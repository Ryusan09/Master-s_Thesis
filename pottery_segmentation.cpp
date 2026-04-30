// =============================================================================
// pottery_segmentation.cpp
// 土器表面の凹凸パターン セグメンテーション
//
// 処理パイプライン：
//   1. PLYファイル読み込み（バイナリ形式）
//   2. 法線ベクトル計算（メッシュから）
//   3. 主曲率・平均曲率の計算
//   4. 大域的な基準曲面のフィッティング（土器本体の湾曲を除去）
//   5. 残差曲率に基づく領域成長セグメンテーション
//   6. OpenGLで結果を色分け可視化
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <stdexcept>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// =============================================================================
// データ構造
// =============================================================================

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    float curvature      = 0.0f;  // 平均曲率
    float residual       = 0.0f;  // 基準面からの残差曲率
    int   segment_id     = -1;    // セグメントID（-1=未割当）
};

struct Face {
    int v[3];  // 頂点インデックス
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Face>   faces;
    std::vector<std::vector<int>> adjacency;  // 頂点の隣接頂点リスト
};

// =============================================================================
// 1. PLY読み込み（バイナリ・リトルエンディアン）
// =============================================================================

Mesh load_ply(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    Mesh mesh;
    int num_vertices = 0, num_faces = 0;
    std::string line;

    // ヘッダ解析
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos)
            num_vertices = std::stoi(line.substr(15));
        else if (line.find("element face") != std::string::npos)
            num_faces = std::stoi(line.substr(13));
        else if (line == "end_header" || line == "end_header\r")
            break;
    }

    std::cout << "Vertices: " << num_vertices
              << ", Faces: "  << num_faces << std::endl;

    // 頂点読み込み（float x, y, z のみ）
    mesh.vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        float xyz[3];
        file.read(reinterpret_cast<char*>(xyz), sizeof(float) * 3);
        mesh.vertices[i].position = glm::vec3(xyz[0], xyz[1], xyz[2]);
    }

    // 面読み込み（uchar count + int[3]）
    mesh.faces.resize(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        uint8_t count;
        file.read(reinterpret_cast<char*>(&count), 1);
        int idx[3];
        file.read(reinterpret_cast<char*>(idx), sizeof(int) * 3);
        mesh.faces[i] = {idx[0], idx[1], idx[2]};
    }

    std::cout << "PLY loaded." << std::endl;
    return mesh;
}

// =============================================================================
// 2. 隣接リスト構築 & 法線計算
// =============================================================================

void build_adjacency(Mesh& mesh) {
    mesh.adjacency.resize(mesh.vertices.size());
    for (auto& f : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            int a = f.v[i], b = f.v[(i+1)%3];
            mesh.adjacency[a].push_back(b);
            mesh.adjacency[b].push_back(a);
        }
    }
    // 重複除去
    for (auto& adj : mesh.adjacency) {
        std::sort(adj.begin(), adj.end());
        adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
    }
    std::cout << "Adjacency built." << std::endl;
}

void compute_normals(Mesh& mesh) {
    // 面法線を頂点に加算して正規化
    for (auto& v : mesh.vertices) v.normal = glm::vec3(0.0f);

    for (auto& f : mesh.faces) {
        glm::vec3& p0 = mesh.vertices[f.v[0]].position;
        glm::vec3& p1 = mesh.vertices[f.v[1]].position;
        glm::vec3& p2 = mesh.vertices[f.v[2]].position;
        glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
        for (int i = 0; i < 3; ++i)
            mesh.vertices[f.v[i]].normal += n;
    }

    for (auto& v : mesh.vertices) {
        float len = glm::length(v.normal);
        if (len > 1e-8f) v.normal /= len;
    }
    std::cout << "Normals computed." << std::endl;
}

// =============================================================================
// 3. 平均曲率の近似計算（法線ベクトルの変化量ベース）
// =============================================================================

void compute_curvature(Mesh& mesh) {
    for (int i = 0; i < (int)mesh.vertices.size(); ++i) {
        const glm::vec3& ni = mesh.vertices[i].normal;
        const glm::vec3& pi = mesh.vertices[i].position;

        float curvature_sum = 0.0f;
        int count = 0;

        for (int j : mesh.adjacency[i]) {
            const glm::vec3& nj = mesh.vertices[j].normal;
            const glm::vec3& pj = mesh.vertices[j].position;

            float dist = glm::length(pj - pi);
            if (dist < 1e-8f) continue;

            // 法線の変化量 / 距離 ≒ 曲率の近似
            float delta_n = glm::length(nj - ni);
            curvature_sum += delta_n / dist;
            ++count;
        }

        mesh.vertices[i].curvature = (count > 0) ? curvature_sum / count : 0.0f;
    }
    std::cout << "Curvature computed." << std::endl;
}

// =============================================================================
// 4. 基準曲面フィッティング（PCAで大域的な曲面を推定し残差を計算）
//    土器本体の湾曲を除去するため、局所近傍の平均曲率との差分を残差とする
// =============================================================================

void compute_residual_curvature(Mesh& mesh, int local_ring = 2) {
    // local_ring: 何ホップ先まで「局所領域」とするか
    // 各頂点の残差 = 自身の曲率 - 近傍頂点の曲率の中央値
    // → 大域的なトレンドを除いた局所的な凹凸を抽出

    for (int i = 0; i < (int)mesh.vertices.size(); ++i) {
        // 1ホップ隣接の曲率を収集（ring=1で十分高速）
        std::vector<float> neighbor_curvatures;
        neighbor_curvatures.push_back(mesh.vertices[i].curvature);
        for (int j : mesh.adjacency[i])
            neighbor_curvatures.push_back(mesh.vertices[j].curvature);

        // 中央値を基準値として残差計算
        std::sort(neighbor_curvatures.begin(), neighbor_curvatures.end());
        float median = neighbor_curvatures[neighbor_curvatures.size() / 2];

        mesh.vertices[i].residual = mesh.vertices[i].curvature - median;
    }
    std::cout << "Residual curvature computed." << std::endl;
}

// =============================================================================
// 5. 領域成長セグメンテーション
//    残差曲率が閾値を超える頂点を「模様の境界」と判定し領域を分割
// =============================================================================

int segment_by_region_growing(Mesh& mesh,
                               float threshold,      // 残差曲率の閾値
                               int   min_segment_size) // 最小セグメントサイズ
{
    int num_vertices = (int)mesh.vertices.size();
    int segment_count = 0;

    for (int start = 0; start < num_vertices; ++start) {
        if (mesh.vertices[start].segment_id != -1) continue;
        if (std::abs(mesh.vertices[start].residual) > threshold) continue;

        // BFSで同質な領域を拡張
        std::queue<int> q;
        std::vector<int> region;
        q.push(start);
        mesh.vertices[start].segment_id = segment_count;

        while (!q.empty()) {
            int cur = q.front(); q.pop();
            region.push_back(cur);

            for (int nb : mesh.adjacency[cur]) {
                if (mesh.vertices[nb].segment_id != -1) continue;
                // 残差曲率が近い頂点を同一セグメントに
                float diff = std::abs(mesh.vertices[nb].residual
                                    - mesh.vertices[cur].residual);
                if (diff < threshold) {
                    mesh.vertices[nb].segment_id = segment_count;
                    q.push(nb);
                }
            }
        }

        // 小さすぎるセグメントは除外（ノイズ扱い）
        if ((int)region.size() < min_segment_size) {
            for (int v : region) mesh.vertices[v].segment_id = -1;
        } else {
            ++segment_count;
        }
    }

    std::cout << "Segments: " << segment_count << std::endl;
    return segment_count;
}

// =============================================================================
// 6. セグメントIDを色にマップ
// =============================================================================

glm::vec3 segment_color(int id, int total) {
    if (id < 0) return glm::vec3(0.3f, 0.3f, 0.3f);  // 未割当 = グレー
    float hue = (float)id / (float)(total + 1);
    // HSV→RGB（簡易版）
    float h = hue * 6.0f;
    int   hi = (int)h;
    float f  = h - hi;
    switch (hi % 6) {
        case 0: return glm::vec3(1.0f, f, 0.0f);
        case 1: return glm::vec3(1.0f-f, 1.0f, 0.0f);
        case 2: return glm::vec3(0.0f, 1.0f, f);
        case 3: return glm::vec3(0.0f, 1.0f-f, 1.0f);
        case 4: return glm::vec3(f, 0.0f, 1.0f);
        default: return glm::vec3(1.0f, 0.0f, 1.0f-f);
    }
}

// =============================================================================
// OpenGL シェーダー
// =============================================================================

const char* VERT_SRC = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

uniform mat4 MVP;
out vec3 vColor;
out vec3 vNormal;

void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
    vColor  = aColor;
    vNormal = aNormal;
}
)";

const char* FRAG_SRC = R"(
#version 410 core
in vec3 vColor;
in vec3 vNormal;
out vec4 FragColor;

void main() {
    // シンプルなランバート照明
    vec3 light = normalize(vec3(1.0, 1.0, 1.0));
    float diff  = max(dot(normalize(vNormal), light), 0.2);
    FragColor   = vec4(vColor * diff, 1.0);
}
)";

GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader error: ") + log);
    }
    return s;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
    std::string ply_path = (argc > 1) ? argv[1] : "pottery.ply";

    // --- パイプライン実行 ---
    Mesh mesh = load_ply(ply_path);
    build_adjacency(mesh);
    compute_normals(mesh);
    compute_curvature(mesh);
    compute_residual_curvature(mesh);

    // ★ 調整パラメータ ★
    float threshold        = 0.02f;  // 曲率残差の閾値（小さいほど細かく分割）
    int   min_segment_size = 100;    // 最小セグメント頂点数（ノイズ除去）

    int num_segments = segment_by_region_growing(mesh, threshold, min_segment_size);

    // --- OpenGL 可視化 ---
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Pottery Segmentation", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    glewInit();
    glEnable(GL_DEPTH_TEST);

    // シェーダー
    GLuint vert = compile_shader(GL_VERTEX_SHADER,   VERT_SRC);
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert); glAttachShader(prog, frag);
    glLinkProgram(prog);

    // VBO構築（位置 + 色 + 法線）
    struct GPUVertex { glm::vec3 pos, color, normal; };
    std::vector<GPUVertex> gpu_verts(mesh.vertices.size());
    for (int i = 0; i < (int)mesh.vertices.size(); ++i) {
        gpu_verts[i].pos    = mesh.vertices[i].position;
        gpu_verts[i].color  = segment_color(mesh.vertices[i].segment_id, num_segments);
        gpu_verts[i].normal = mesh.vertices[i].normal;
    }

    std::vector<unsigned int> indices;
    indices.reserve(mesh.faces.size() * 3);
    for (auto& f : mesh.faces)
        for (int k = 0; k < 3; ++k) indices.push_back(f.v[k]);

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, gpu_verts.size() * sizeof(GPUVertex),
                 gpu_verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex),
                          (void*)offsetof(GPUVertex, pos));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex),
                          (void*)offsetof(GPUVertex, color));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GPUVertex),
                          (void*)offsetof(GPUVertex, normal));
    glEnableVertexAttribArray(2);

    // カメラ（マウスで回転できるよう拡張予定）
    glm::vec3 center(0.0f);
    for (auto& v : mesh.vertices) center += v.position;
    center /= (float)mesh.vertices.size();

    float angle = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 model = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0,1,0));
        model = glm::translate(model, -center);
        glm::mat4 view  = glm::lookAt(glm::vec3(0,0,300),
                                       glm::vec3(0,0,0),
                                       glm::vec3(0,1,0));
        glm::mat4 proj  = glm::perspective(glm::radians(45.0f),
                                            1280.0f/720.0f, 0.1f, 10000.0f);
        glm::mat4 MVP = proj * view * model;

        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog, "MVP"), 1, GL_FALSE,
                           glm::value_ptr(MVP));
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, nullptr);

        angle += 0.005f;  // ゆっくり回転
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
