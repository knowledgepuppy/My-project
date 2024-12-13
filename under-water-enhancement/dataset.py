from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化水下图像数据集
        
        Args:
            root_dir: 数据集根目录路径
            transform: 图像转换操作
        """
        # 直接指向images目录
        self.images_dir = Path(root_dir) / 'images'
        self.transform = transform
        
        print(f"\n初始化数据集...")
        print(f"图片目录: {self.images_dir}")
        
        # 只搜索.jpg文件
        self.images = list(self.images_dir.glob('*.jpg'))
        print(f"找到 {len(self.images)} 张JPG图片")
        
        if len(self.images) == 0:
            raise RuntimeError("没有找到任何图片文件")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image