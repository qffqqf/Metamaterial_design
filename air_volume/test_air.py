import os
from ngsolve import *
from netgen.meshing import IdentificationType
from netgen.read_gmsh import ReadGmsh
import numpy as np

# --- 1. 参数设置 ---
L, W = 136e-3, 139e-3
H = 30e-3
freq = 1000  # 频率 (Hz)
c0 = 343.0   # 声速 (m/s)
k0 = 2 * pi * freq / c0

# Bloch-Floquet 波矢 (kx, ky, 0) - 这里可以根据你的频散分析需求修改
kx_BF = 10.0
ky_BF = 5.0
k_BF = CF((kx_BF, ky_BF, 0))

# 高斯点声源参数
# 几何中心计算：X ∈ [0, L], Y ∈ [0, W], Z ∈ [-H/3, 2H/3]
xc, yc, zc = L / 2, W / 2, H / 6
sigma = 0.005  # 高斯声源宽度
A = 1.0        # 声源幅值

# --- 2. 读取网格与边界识别 ---
mesh_path = "./tutorials/mesh_files/t8.msh"
if not os.path.exists(mesh_path):
    raise FileNotFoundError(f"未找到网格文件: {mesh_path}")

# 读取网格
mesh = Mesh(mesh_path)

# 识别周期性边界
# 注意：Gmsh 生成非结构化网格时，面上节点如果未强制匹配，NGSolve 可能会在识别时抛出警告。
# 确保在 Gmsh 中使用了 setPeriodic，或者在物理场简单的规则几何上能自动匹配。
mesh.ngmesh.IdentifyRegion("Boundary_X_Min", "Boundary_X_Max", IdentificationType.PERIODIC)
mesh.ngmesh.IdentifyRegion("Boundary_Y_Min", "Boundary_Y_Max", IdentificationType.PERIODIC)

# --- 3. 定义有限元空间 ---
# 使用复数空间求解声学问题
fes = H1(mesh, order=3, complex=True)

# 试探函数和测试函数（此处 u, v 为去除 Bloch 相位后的包络函数）
u, v = fes.TnT()

# --- 4. 建立 Bloch-Floquet 弱形式 ---
# 在 Bloch-Floquet 理论中，实际声场 U = u * exp(-i * k_BF * r)
# 弱形式算子从 ∇ 变为 (∇ - i * k_BF)
def grad_BF(w):
    return grad(w) - 1j * k_BF * w

a = BilinearForm(fes)
# (∇ - ik)u · (∇ - ik)v - k0^2 u v
a += (InnerProduct(grad_BF(u), grad_BF(v)) - k0**2 * u * Conj(v)) * dx

with TaskManager():
    a.Assemble()

# --- 5. 设置高斯点声源 (RHS) ---
r2 = (x - xc)**2 + (y - yc)**2 + (z - zc)**2
# 高斯分布包络
source_expr = A * exp(-r2 / (2 * sigma**2))

f = LinearForm(fes)
# 为了使得右端项与测试空间匹配，需要补偿相位的共轭部分
f += source_expr * exp(1j * (kx_BF * x + ky_BF * y)) * Conj(v) * dx

with TaskManager():
    f.Assemble()

# --- 6. 求解有限元方程 ---
gfu = GridFunction(fes, name="envelope")

print("开始求解 Bloch-Floquet 周期性声场...")
with TaskManager():
    # 求解代数方程组
    inv = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack")
    gfu.vec.data = inv * f.vec
print("求解完成。")

# --- 7. 后处理与可视化 ---
# 恢复真实的物理声场 U(x) = u(x) * exp(-i * k_BF * x)
true_acoustic_field = gfu * exp(-1j * (kx_BF * x + ky_BF * y))

# 将真实声场插值到新的 GridFunction 中以便输出或通过 Netgen GUI 查看
gfu_real = GridFunction(fes, name="true_acoustic_field")
gfu_real.Set(true_acoustic_field)

# 使用 VTK 输出以便在 PyVista 或 ParaView 中进行分析
vtk = VTKOutput(ma=mesh,
                coefs=[gfu_real.real, gfu_real.imag, Abs(gfu_real)],
                names=["real_pressure", "imag_pressure", "abs_pressure"],
                filename="acoustic_results",
                subdivision=2)
vtk.Do()

print("结果已保存至 acoustic_results.vtu。可以使用 PyVista 或 ParaView 进行可视化。")