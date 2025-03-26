import glob
from datetime import time
from text2vec import SentenceModel
from tqdm import tqdm
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import chromadb
import os
import pdfplumber
import docx
import pandas as pd
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from PIL.ExifTags import TAGS
import torch

# https://huggingface.co/shibing624/text2vec-base-chinese #通用
# https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase #语义匹配会更好一点


def extract_text_from_file(file_path):
    """
    根据文件类型提取文本内容。
    支持 .txt、.pdf、.docx、.epub、.csv、.xlsx 格式。
    """
    ext = file_path.split('.')[-1].lower()
    text = ''
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    elif ext == 'docx':
        doc = docx.Document(file_path)
        # 提取正文内容
        for para in doc.paragraphs:
            text += para.text + '\n'
        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + '\n'
    elif ext == 'epub':
        book = epub.read_epub(file_path)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # 使用BeautifulSoup解析HTML内容
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + '\n'
    elif ext == 'csv':
        df = pd.read_csv(file_path)
        text = df.to_string()
    elif ext == 'xlsx':
        df = pd.read_excel(file_path)
        text = df.to_string()
    return text


def chunk_text(text, chunk_size=500):
    """
    将文本分块，每个块的最大长度为 chunk_size。
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    print(chunks)
    return chunks


def process_files_in_directory(files):
    """
    遍历目录中的所有文件，提取文本并将其分块后存储到 chromadb。
    每个文件的元数据包含文件路径和原始文本内容。
    """
    # 初始化Chroma和文本模型
    model = SentenceModel('./text2vec-base-chinese-paraphrase')
    db_path = r"./vector_db"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name='text2vec')
    # 遍历文件并处理
    for file in tqdm(files, desc="处理文件"):
        text = extract_text_from_file(file)
        print(f"处理文件：{file}, 提取文本长度：{len(text)}")  # 打印文件名和文本长度
        if text.strip():
            # 对每个文件进行文本分块
            file_chunks = chunk_text(text)
            # 将每个文件的文本块及其 embeddings 存入 chromadb
            for idx, chunk in enumerate(file_chunks):
                metadata = {
                    'file_path': file,  # 文件路径
                    'original_info': chunk  # 原始文本内容
                }
                print(metadata)
                ids = str(file) + "-" + str(idx)
                collection.add(ids=ids,
                               metadatas=[metadata],
                               embeddings=model.encode(chunk))
                print(ids, chunk)  # 用于调试

    print("所有文本块已成功存入数据库！")


def process_imgs_in_directory(imgs):
    """
    遍历指定文件夹中的图像，生成嵌入并添加到集合。
    """
    db_path = r"./vector_db"
    client = chromadb.PersistentClient(path=db_path)
    # 直接在线加载Hugging Face上的中文CLIP模型
    # model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    # clip_model = ChineseCLIPModel.from_pretrained(model_name)
    # clip_processor = ChineseCLIPProcessor.from_pretrained(model_name)
    # 加载本地已经下载好的模型，原始链接：https://huggingface.co/docs/transformers/model_doc/chinese_clip
    # 初次使用需要将https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16，模型下载到本地。
    model_name = "./chinese-clip-vit-base-patch16"  # 包含已训练的模型的结构和参数
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(
        model_name)  # 将图像转换为模型可以接受的格式（如张量），并进行预处理。
    # 获取或创建集合
    collection = client.get_or_create_collection(name='img2vec')
    # 将找到的图片向量化
    for image_path in tqdm(imgs, desc="图片向量化至数据库的进度"):
        try:
            image = Image.open(image_path).convert("RGB")  # 确保所有图片转换为RGB格式
            # 尝试获取 EXIF 数据作为metadata
            exif_data = image.getexif()
            if not exif_data:
                exif_str = "No EXIF data found."
            else:
                exif_info = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)  # 获取 EXIF 标签名称
                    exif_info[tag_name] = value
                # 将 EXIF 信息转换为字符串
                exif_str = "\n".join(
                    [f"{key}: {value}" for key, value in exif_info.items()])

            # 定义嵌入函数，接收图像并生成嵌入。
            inputs = processor(
                images=image, return_tensors="pt", padding=True
            )  # 将图像预处理为PyTorch 张量，同时允许输入数据在尺寸不一致时自动填充，保证模型能够正确处理。
            outputs = model.get_image_features(**inputs).detach().numpy(
            )  # 计算并返回图像的特征向量，这些向量是图像在模型的特征空间中的表示，可以用来做图像匹配、检索等任务。detach表示不进行梯度计算，numpy将PyTorch张量转换为numpy数组，方便后续在CPU设备上进行运算。
            # 添加嵌入到集合
            metadata = {
                'file_path': image_path,  # 文件路径
                'original_info': exif_str  # 原始exif
            }
            print(metadata)
            collection.add(ids=image_path,
                           metadatas=[metadata],
                           embeddings=outputs)
            # 这边记录一个实际写入向量数据库的数据格式的案例，实际生产要结合业务定义字段。
            # {
            #     "id": "image_001",
            #     "embedding": [0.245, 0.788, ..., 0.541],
            #     "metadata": {
            #         "source": "path/to/image.jpg",
            #         "category": "nature",
            #         "tags": ["sunset", "sky"],
            #         "created_at": "2025-01-14T12:00:00"
            #     }
            # }

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def query_vec(query):
    """
    输入一段话，从 chromadb 数据库中找出最相近的三个 chunk。
    """
    # 初始化客户端和集合
    db_path = r"./vector_db"
    client = chromadb.PersistentClient(path=db_path)
    collection_text = client.get_or_create_collection(name='text2vec')
    collection_img = client.get_or_create_collection(name='img2vec')

    # 查询近似度前三的文本chunks
    model_text = SentenceModel('./text2vec-base-chinese-paraphrase')
    query_embeddings_text = model_text.encode(query)
    results_text = collection_text.query(
        query_embeddings=query_embeddings_text,
        n_results=2,
        include=["metadatas", "distances"])
    # 查询近似度前三的图片
    clip_model = ChineseCLIPModel.from_pretrained(
        "./chinese-clip-vit-base-patch16")
    clip_processor = ChineseCLIPProcessor.from_pretrained(
        "./chinese-clip-vit-base-patch16")
    inputs = clip_processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(
            **inputs).cpu().numpy()  # 将结果移回CPU并转换为NumPy
    query_embeddings_img = outputs.tolist()  # 返回 List[List[float]]
    results_img = collection_img.query(query_embeddings=query_embeddings_img,
                                       n_results=2,
                                       include=["metadatas", "distances"])
    return [results_text, results_img]


def vector_all():
    directory = './files'
    finished_files_path = './vector_db/finished_files'
    # 获取所有文件路径
    all_files = glob.glob(os.path.join(directory, '**', '*'), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]

    # 读取已完成的文件列表
    if os.path.exists(finished_files_path):
        with open(finished_files_path, 'r', encoding='utf-8') as f:
            finished_files = set(f.read().splitlines())  # 读取已完成文件，存入集合（去重）
    else:
        finished_files = set()  # 如果文件不存在，则为空集合

    # 找出未完成的文件
    unfinished_files = [f for f in all_files if f not in finished_files]
    # 图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    # 文本格式
    text_extensions = {'.txt', '.pdf', '.docx', '.epub', '.csv', '.xlsx'}

    # 分离未完成的图片和文本文件
    unfinished_imgs = [
        f for f in unfinished_files
        if os.path.splitext(f)[-1].lower() in image_extensions
    ]
    unfinished_files = [
        f for f in unfinished_files
        if os.path.splitext(f)[-1].lower() in text_extensions
    ]

    # 处理所有的文件
    process_files_in_directory(unfinished_files)
    process_imgs_in_directory(unfinished_imgs)

    # 写入新的 finished_files
    with open(finished_files_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_files))  # 更新为新的 all_files 列表


if __name__ == "__main__":
    vector_all()
