import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

class ImageLoader:
    """
    图片加载器类，用于加载单张图片或整个文件夹中的图片，并提取EXIF元数据
    
    功能：
    1. 加载单张图片或整个文件夹中的图片
    2. 提取基本图片信息（尺寸、格式等）
    3. 提取完整的EXIF元数据
    4. 提取GPS位置信息（如果存在）
    """
    def __init__(self, path, load_content=False):
        """
        初始化图片加载器
        
        参数:
        path (str): 图片文件路径或包含图片的文件夹路径
        load_content (bool): 是否加载图片内容到内存
        """
        self.image_data = []
        self.load_content = load_content
        
        if os.path.isfile(path):
            self._load_single_image(path)
        elif os.path.isdir(path):
            self._load_folder_images(path)
        else:
            raise ValueError(f"路径不存在或不是有效的文件/文件夹: {path}")
    
    def _load_single_image(self, image_path):
        """加载单张图片元数据，可选加载内容"""
        try:
            # 使用with语句确保文件正确关闭
            with Image.open(image_path) as img:
                # 获取基本图片信息
                image_info = {
                    "path": image_path,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size": os.path.getsize(image_path),
                    "exif": self._get_exif_data(img),
                    "gps": self._get_gps_info(self._get_exif_data(img))
                }
                
                # 只在需要时存储图片内容
                if self.load_content:
                    image_info["image"] = img.copy()
                
                self.image_data.append(image_info)
        except Exception as e:
            print(f"加载图片 {image_path} 时出错: {str(e)}")
    
    def _load_folder_images(self, folder_path):
        """加载文件夹中的所有图片元数据"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
                self._load_single_image(file_path)
    
    def _get_exif_data(self, image):
        """提取并解析图片的EXIF数据"""
        exif_data = {}
        try:
            info = image._getexif()
            if info:
                for tag, value in info.items():
                    decoded = TAGS.get(tag, tag)
                    exif_data[decoded] = value
        except Exception as e:
            print(f"提取EXIF数据时出错: {str(e)}")
        return exif_data
    
    def _get_gps_info(self, exif_data):
        """提取并解析GPS信息（如果存在）"""
        gps_info = {}
        if 'GPSInfo' in exif_data:
            for key in exif_data['GPSInfo'].keys():
                decoded_key = GPSTAGS.get(key, key)
                gps_info[decoded_key] = exif_data['GPSInfo'][key]
        return gps_info
    
    def get_image_data(self):
        """获取所有加载图片的数据"""
        return self.image_data
    
    def get_image_count(self):
        """获取加载的图片数量"""
        return len(self.image_data)
    
    def get_image_by_index(self, index):
        """通过索引获取图片对象"""
        if 0 <= index < len(self.images):
            return self.images[index]
        return None
    
    def get_image_data_by_index(self, index):
        """通过索引获取图片数据"""
        if 0 <= index < len(self.image_data):
            return self.image_data[index]
        return None
    
    def get_image_content(self, index):
        """按需获取图片内容"""
        if 0 <= index < len(self.image_data):
            img_data = self.image_data[index]
            if "image" in img_data:
                return img_data["image"]
            
            # 如果未加载，则按需加载
            try:
                with Image.open(img_data["path"]) as img:
                    return img.copy()
            except Exception as e:
                print(f"加载图片内容时出错: {str(e)}")
        return None

    def close_all(self):
        """关闭所有图片文件"""
        for img in self.images:
            img.close()
        self.images = []