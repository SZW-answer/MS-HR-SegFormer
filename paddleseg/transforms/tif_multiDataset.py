from osgeo import gdal
import numpy as np
import cv2

def read_multiband_tif(filepath):
    # 使用 'gdal' 打开文件
    dataset = gdal.Open(filepath)

    # 读取全部波段
    data = dataset.ReadAsArray()  # 这会读取所有波段到一个 NumPy 数组中

    # 将数据转换为float32格式并调整维度顺序
    data = data.astype(np.float32).transpose((1, 2, 0))

    return data
def save_as_png(image_array, file_path):
    """
    Save a (256, 256, 3) Numpy array as a PNG image.

    Parameters:
    - image_array: A (256, 256, 3) Numpy array representing image data.
    - file_path: A string with the file path where the image will be saved.
    
    Returns:
    - A boolean value: True if the image was saved successfully, False otherwise.
    """
    # 检查传入的数组是否是预期的形状
    # if image_array.shape != (256, 256, 1):
    #     raise ValueError("The input array must have the shape (256, 256, 3)")

    # 检查数组中的数据类型是否为 uint8, 如果不是，需要转换
    if image_array.dtype != np.uint8:
        print("Warning: converting image array to uint8.")
        image_array = image_array.astype(np.uint8)

    # 写入文件，保存为PNG
    return cv2.imwrite(file_path, image_array, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# 用法示例，创建一个简单的测试数组
if __name__ == '__main__':
    g = read_multiband_tif("C:\\Users\\GDOS\\Desktop\\PaddleSeg-release-2.8 - szw\\dataset\\labels\\000000030775.tif")
    # data = cv2.imread("C:\\Users\\GDOS\\Desktop\\PaddleSeg-release-2.8 - szw\\dataset\labels\\000000030774.tif").astype('float32')
    print(g)
    # print(data.shape)
    # print(data-g)
    save_as_png(g,"gdal.png")
  
    # save_as_png(data-g,"diff.png")
    