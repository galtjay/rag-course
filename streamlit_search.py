import streamlit as st
from embed_files import query_vec
from PIL import Image 
from duckduckgo import rag_search


def format_distance(distance):
    """Formats the distance score as a percentage."""
    similarity = max(0, min(100, (1 - distance / 2000) * 100))
    return f"{similarity:.1f}%"


def is_image_file(file_path):
    """Checks if a file is likely an image using PIL."""
    try:
        Image.open(file_path)
        return True
    except (FileNotFoundError, IOError,
            Image.UnidentifiedImageError):  # Catch specific errors
        return False


def main():
    st.set_page_config(page_title="云手机语义搜索", page_icon="🔍", layout="wide")

    st.markdown(
        """        <style>
        /* 整体样式 */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* 搜索区域样式 */
        .search-container {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* 结果卡片样式 */
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* 相似度标签样式 */
        .similarity-badge {
            background-color: #007bff;
            color: white;
            padding: 0.3rem 0.6rem;
            border-radius: 15px;
            font-size: 0.9rem;
            margin-left: 0.5rem;
        }

        /* 文件路径样式 */
        .file-path {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9rem;
            margin: 0.5rem 0;
            word-break: break-all;
        }

        /* 内容区域样式 */
        .content-area {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            overflow: auto; /* Add scroll for long content */
            font-size: 0.85rem; /* Reduced font size */
        }

        /* 图片容器 */
        .image-container {
            position: relative;
            width: 100%;
            border-radius: 10px;
            overflow: hidden; /* Ensure image doesn't overflow */
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Add shadow */
         }

        .image-container img {
            width: 100%;
            height: auto;  /* Maintain aspect ratio */
            display: block;  /* Remove extra space below image */
        }

        /* 分隔线样式 */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 1px solid #e0e0e0;
        }

        /* 互联网搜索结果样式 */
        .web-result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #4285f4; /* Google蓝色边框 */
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .web-result-title {
            color: #1a0dab;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .web-result-url {
            color: #006621;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            word-break: break-all;
        }
        
        .web-result-snippet {
            color: #545454;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* 来源标签 */
        .source-badge {
            background-color: #4285f4;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }

        /* 移动端适配 */
        @media (max-width: 768px) {
            .result-card, .web-result-card {
                padding: 1rem;
            }
            .search-container {
                padding: 1rem;
            }
            .image-container{
              width: 100%;
            }

            .image-container img {
                width: 100%;
                height: auto;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='text-align: center; '>📱 云手机语义搜索</h1>",
                unsafe_allow_html=True)

    # 使用会话状态来存储查询
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    # 处理表单提交
    with st.form(key='search_form'):
        query = st.text_input("搜索内容",
                              placeholder="请输入搜索关键词...",
                              help="支持语义搜索，可以输入描述性的语句",
                              value=st.session_state.search_query,
                              label_visibility="collapsed")  # 或者用 "hidden"

        search_button = st.form_submit_button("🔍 搜索",
                                              type="primary",
                                              use_container_width=True)

        if search_button:
            st.session_state.search_query = query

    # 如果有查询且不为空，则执行搜索
    if st.session_state.search_query.strip():
        with st.spinner("🔍 正在RAG搜索:包含互联网、文档语义、图片语义"):
            # 执行互联网搜索
            internet_results = rag_search(st.session_state.search_query)

            # 获取本地文档搜索结果
            result = query_vec(st.session_state.search_query)

            # 显示互联网搜索结果
            if internet_results:
                st.markdown("### 🌐 互联网搜索结果")
                st.markdown(f"""<div class='web-result-card'>
                        <div class='web-result-snippet'>{internet_results}</div>
                    </div>""",
                            unsafe_allow_html=True)

            # Text Results
            result_texts = result[0]
            if result_texts and result_texts.get("metadatas"):
                st.markdown("### 📄 文档结果")
                for i, (metadata, distance) in enumerate(
                        zip(result_texts["metadatas"][0],
                            result_texts["distances"][0])):
                    similarity = format_distance(distance)
                    st.markdown(
                        f"""<div class='result-card'><div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'><h4 style='margin: 0;'>相关文档 {i + 1}</h4><span class='similarity-badge'>相关度: {similarity}</span></div><div class='file-path'>{metadata.get('file_path', '未知文件')}</div><div class='content-area'>{metadata.get('original_info', '无内容')}</div></div>""",
                        unsafe_allow_html=True,
                    )
            # Image Results
            result_images = result[1]
            if result_images and result_images.get("metadatas"):
                st.markdown("### 🖼️ 图片结果")
                # Use a single row with columns for better responsiveness
                image_cols = st.columns(2)  # Display images in two columns

                for i, (metadata, distance) in enumerate(
                        zip(result_images["metadatas"][0],
                            result_images["distances"][0])):
                    similarity = format_distance(distance)
                    file_path = metadata.get("file_path", "未知图片")
                    col = image_cols[i % 2]  # Alternate between columns

                    with col:
                        st.markdown(
                            f"""
                            <div class='result-card'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                                    <h4 style='margin: 0;'>相关图片 {i + 1}</h4>
                                    <span class='similarity-badge'>相关度: {similarity}</span>
                                </div>
                                """,
                            unsafe_allow_html=True,
                        )

                        if is_image_file(file_path):  # Check if it's an image
                            try:
                                # Read image and display
                                with open(file_path, "rb") as image_file:
                                    image_bytes = image_file.read()
                                st.markdown("<div class='image-container'>",
                                            unsafe_allow_html=True)
                                st.image(image_bytes,
                                         caption=None,
                                         use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                            except Exception as e:
                                st.error(
                                    f"Error loading image {file_path}: {e}")
                                st.markdown(
                                    f"<div class='file-path'>无法显示图片：{file_path}</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f"<div class='file-path'>无法显示图片：{file_path} (Not a recognized image file)</div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f"""
                                <div class='file-path'>{file_path}</div>
                                <div class='content-area'>
                                    {metadata.get('original_info', '无 EXIF 数据')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
