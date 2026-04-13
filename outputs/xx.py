import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def generate_palette_image(dataset_name, color_map, save_dir="./palettes"):
    """
    根据给定的 color_map 生成该数据集的色块展示图。
    图片按节点类型数量等分。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    node_types = list(color_map.keys())
    colors = list(color_map.values())
    num_types = len(node_types)

    if num_types == 0:
        print(f"Dataset {dataset_name} has no node types.")
        return

    # 创建画布
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)  # 尽量铺满

    # 计算每个色块的高度
    block_height = 1.0 / num_types

    # 循环绘制色块和文本
    for i, (ntype, color) in enumerate(zip(node_types, colors)):
        # 计算起始纵坐标 (从上往下画)
        y_start = 1.0 - (i + 1) * block_height

        # 绘制色块矩形
        rect = patches.Rectangle((0, y_start), 1.0, block_height,
                                 linewidth=0, facecolor=color, alpha=0.9)
        ax.add_patch(rect)

        # 添加文本描述 (节点类型名 + 十六进制色值)
        # 尝试使用和谐的文本颜色：如果颜色过浅，用黑色；如果颜色过深，用白色
        r, g, b = plt.cm.colors.to_rgb(color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = 'white' if luminance < 0.5 else 'black'

        text_label = f"Node Type: {ntype}\nColor: {color}"
        ax.text(0.5, y_start + block_height / 2.0, text_label,
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, fontweight='bold', color=text_color, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', boxstyle='round'))

    # 设置坐标轴属性
    ax.set_axis_off()  # 隐藏坐标轴
    plt.title(f"Harmony Palette for Dataset: {dataset_name}", fontsize=24, fontweight='bold', pad=20)

    # 保存图片
    save_path = f"{save_dir}/{dataset_name}_palette.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Palette image saved to: {save_path}")


if __name__ == "__main__":
    # --- 1. 使用你当前的颜色（用于对比测试） ---
    # lastfm_colors = {'artist': 'skyblue', 'user': 'lightgreen'}
    # aug_citation_colors = {'author': 'skyblue', 'fos': 'lightgreen', 'paper': 'lightcoral', 'ref': 'salmon'}
    # acm_colors = {'author': 'skyblue', 'field': 'lightgreen', 'paper': 'lightcoral'}

    # --- 2. 使用推荐的专业和谐色系 ---
    # lastfm_colors_safe = {'artist': 'skyblue', 'user': 'lightgreen'}
    # aug_citation_colors_safe = {'author': 'skyblue', 'fos': 'lightgreen', 'paper': 'lightcoral', 'ref': 'lightcoral'}
    # acm_colors_safe = {'author': 'skyblue', 'field': 'lightgreen', 'paper': 'lightcoral'}
    lastfm_colors_safe = {
        'artist': '#9EBBD7',  # 雾霾天蓝 (Dusty Baby Blue)
        'user': '#A7CFAB'    # 灰调薄荷绿 (Grayish Mint Green)
    }
    aug_citation_colors_safe = {
        'paper': '#FBB4AE',   # 珊瑚粉 (核心)
        'fos': '#CCEBC5',     # 薄荷绿 (领域)
        'author': '#B3CDE3',  # 冰晶蓝 (人物)
        'ref': '#E5E5E5'      # 浅珍珠灰 (辅助/参考文献)
    }
    acm_colors_safe = {
        'author': '#B0C4DE',  # 雾霾淡蓝 (Light Dusty Blue)
        'field': '#A9DFBF',   # 雾霾淡绿 (Light Dusty Green)
        'paper': '#E6B0C1'    # 雾霾粉红 (Light Dusty Pink)
    }

    # 生成图片
    generate_palette_image('lastfm1', lastfm_colors_safe)
    generate_palette_image('aug_citation2', aug_citation_colors_safe)
    generate_palette_image('ACM1', acm_colors_safe)