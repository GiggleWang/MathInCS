import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# 定义创建三角形图像的函数
def create_triangle(size, base, height):
    image = np.ones((size, size)) * 0.99  # 白色背景
    center_x = size // 2
    center_y = size // 2

    # 计算三角形顶点
    v1 = (center_x, center_y)
    v2 = (center_x + base // 2, center_y - height // 2)
    v3 = (center_x - base // 2, center_y - height // 2)

    # 使用 numpy 和逻辑索引绘制三角形
    mask = np.zeros((size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            # 使用 barycentric 坐标验证点 (x, y) 是否在三角形内
            alpha = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / (
                        (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1]))
            beta = ((v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])) / (
                        (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1]))
            gamma = 1.0 - alpha - beta
            if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
                mask[x, y] = True

    # 将三角形区域设置为黑色
    image[mask] = 0.01

    return image


# 定义图像大小和三角形的底和高
size = 100
base = 40
height = 30

# 使用上述函数创建三角形图像
triangle_image = create_triangle(size, base, height)

# 可视化原始图像
plt.imshow(triangle_image, cmap='gray')
plt.title('Original Triangle Image')
plt.savefig('./image/triangle_original.png')
plt.show()
# 构建优化问题
# 假设图像大小是固定的，这里我们使用100x100的图像
n = size * size  # 总像素数

# 定义变量，这里x是二进制的，表示每个像素属于圆圈还是背景
x = cp.Variable(n, boolean=True)

# 将 image 转换为一维数组以便与 x 进行逐元素操作
image_flatten = triangle_image.flatten()

# 第一部分：惩罚错误地将圆圈像素分配为背景
circle_penalty = cp.sum(cp.multiply((1 - x), (0.99 - image_flatten)) ** 2)

# 第二部分：惩罚错误地将背景像素分配为圆圈
background_penalty = cp.sum(cp.multiply(x, image_flatten - 0.01) ** 2)

# 目标函数：最小化两种错误的总和
objective = cp.Minimize(circle_penalty + background_penalty)

# 构建问题并解决
prob = cp.Problem(objective)
prob.solve(solver=cp.ECOS_BB)  # 使用ECOS_BB求解器

segmented_image = np.array(x.value).reshape((size, size))
segmented_image = 1 - segmented_image  # 反转布尔值，1 代表背景，0 代表圆圈
plt.imshow(segmented_image, cmap='gray')
plt.title('triangle_segmented Image')
plt.savefig('./image/triangle_segmented.png')
plt.show()


# 检查结果
print("Status:", prob.status)
print("Optimal value:", prob.value)