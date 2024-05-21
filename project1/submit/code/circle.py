import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# 定义创建图像的函数
def create_image(size, radius):
    image = np.ones((size, size)) * 0.99
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= radius
    image[mask] = 0.01
    return image


# 定义图像大小和圆的半径
size = 100
radius = 30

# 使用上述函数创建图像
image = create_image(size, radius)

# 可视化原始图像
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.savefig('./image/circle_original.png')
plt.show()

# 构建优化问题
# 假设图像大小是固定的，这里我们使用100x100的图像
n = size * size  # 总像素数

# 定义变量，这里x是二进制的，表示每个像素属于圆圈还是背景
x = cp.Variable(n, boolean=True)

# 将 image 转换为一维数组以便与 x 进行逐元素操作
image_flatten = image.flatten()

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
plt.title('circle_segmented Image')
plt.savefig('./image/circle_segmented.png')
plt.show()


# 检查结果
print("Status:", prob.status)
print("Optimal value:", prob.value)