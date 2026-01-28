# app.py
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px  # ← カラーパレット用に追加
import streamlit as st


# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="EctForce Plotter (Filename-colored)", layout="wide")


# -----------------------------
# ユーティリティ
# -----------------------------
def _parse_float_list(line: str) -> List[float]:
    parts = [p.strip() for p in line.split(",")]
    vals = []
    for p in parts:
        if p == "" or p.startswith("//"):
            break
        try:
            vals.append(float(p))
        except ValueError:
            break
    return vals


def _read_start_end_from_line_225(lines: List[str]) -> Tuple[float, float]:
    """
    1-based 225行目（0-based 224）の数列末尾2要素を StartPos, EndPos として取得。
    フォールバック： [MEASURE PARAMETER] 直後の数列。
    """
    if len(lines) >= 225:
        vals = _parse_float_list(lines[224])
        if len(vals) >= 2:
            return float(vals[-2]), float(vals[-1])
    # fallback
    start_idx = None
    for i, line in enumerate(lines):
        if "[MEASURE PARAMETER]" in line:
            start_idx = i
            break
    if start_idx is not None:
        for j in range(start_idx + 1, min(start_idx + 10, len(lines))):
            vals = _parse_float_list(lines[j])
            if len(vals) >= 2:
                return float(vals[-2]), float(vals[-1])
    raise ValueError("StartPos/EndPos を 225行目等から取得できませんでした。")


def _read_table_from_line_245(lines: List[str]) -> pd.DataFrame:
    """
    1-based 245行目をヘッダとして、以降の MntTable,Head,Index,Offset の4列を読む。
    次のセクション([で始まる])や空行まで。
    """
    data_start = 245  # 1-based
    i0 = data_start - 1
    if len(lines) < data_start:
        raise ValueError("ファイルが短すぎて 245 行目に到達できません。")

    header_raw = lines[i0].strip()
    header = header_raw.lstrip("/").lstrip()  # 例: //MntTable, Head, Index, Offset
    header_cols = [c.strip() for c in header.split(",") if c.strip()]
    if len(header_cols) != 4 or header_cols[0].lower().replace(" ", "") not in ("mnttable",):
        header_cols = ["MntTable", "Head", "Index", "Offset"]

    rows = []
    for k in range(i0 + 1, len(lines)):
        line = lines[k].strip()
        if line == "" or line.startswith("["):
            break
        if "//" in line:
            line = line.split("//", 1)[0].strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            mnt = int(parts[0]); head = int(parts[1]); idx = int(parts[2]); off = float(parts[3])
        except ValueError:
            continue
        rows.append((mnt, head, idx, off))

    if not rows:
        raise ValueError("245行目以降に有効なデータ行が見つかりません。")

    df = pd.DataFrame(rows, columns=["MntTable", "Head", "Index", "Offset"])
    return df


def make_x_coordinate(
    df_group: pd.DataFrame,
    start_pos: float,
    end_pos: float,
    index_zero_based: bool = True
) -> pd.Series:
    """横軸：StartPos→EndPos を Index で線形割付。"""
    if df_group.empty:
        return pd.Series(dtype=float)
    idx = df_group["Index"].astype(float)
    if index_zero_based:
        max_idx = idx.max()
        denom = max_idx if max_idx != 0 else 1.0
        t = idx / denom
    else:
        min_idx = idx.min()
        max_idx = idx.max()
        span = max_idx - min_idx if max_idx != min_idx else 1.0
        t = (idx - min_idx) / span
    return start_pos + (end_pos - start_pos) * t


# -----------------------------
# データ構造
# -----------------------------
@dataclass
class FileBundle:
    name: str
    lines: List[str]


# -----------------------------
# Z 候補の抽出（キャッシュ）
# -----------------------------
@st.cache_data(show_spinner=False)
def scan_z_keys_cached(text_payloads: List[str]) -> List[str]:
    z_set = set()
    for text in text_payloads:
        lines = text.splitlines()
        df = _read_table_from_line_245(lines)
        df["TableLabel"] = df["MntTable"].map({0: "A", 1: "B"}).fillna(df["MntTable"].astype(str))
        df["HeadDisp"] = df["Head"] + 1
        for (tbl, hdisp), _ in df.groupby(["TableLabel", "HeadDisp"]):
            z_set.add(f"Z{tbl}{int(hdisp)}")

    def z_key_sort(s: str):
        tbl = s[1] if len(s) >= 2 else "A"
        nums = re.findall(r"\d+", s)
        h = int(nums[0]) if nums else 0
        return (tbl, h)

    return sorted(z_set, key=z_key_sort)


# -----------------------------
# カラーマップ（ファイル名 → color）
# -----------------------------
def make_filename_color_map(file_names: List[str]) -> Dict[str, str]:
    """
    ステーブルな順序（sorted）で色を割当て。
    既定は Plotly の定性カラースキーム（Set3 + Bold を連結、足りなければ循環）。
    """
    palette = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Plotly
    )
    uniq = sorted(dict.fromkeys(file_names))  # 安定化 & 重複排除
    color_map = {}
    n = len(palette)
    for i, fn in enumerate(uniq):
        color_map[fn] = palette[i % n]
    return color_map


# -----------------------------
# フィギュア作成（凡例＝ファイル名 & 色固定）
# -----------------------------
def build_figure(
    bundles: List[FileBundle],
    index_zero_based: bool,
    selected_z: Optional[List[str]]
) -> go.Figure:
    """
    Z キーでフィルタし、各 Z ごとにサブプロット（4列）。
    凡例はファイル名で1回のみ表示。色はファイル名で固定。
    """
    all_entries = []
    z_keys_set = set()

    # ファイル名 → 色
    fname_color = make_filename_color_map([b.name for b in bundles])

    for b in bundles:
        start_pos, end_pos = _read_start_end_from_line_225(b.lines)
        df = _read_table_from_line_245(b.lines)
        df["TableLabel"] = df["MntTable"].map({0: "A", 1: "B"}).fillna(df["MntTable"].astype(str))
        df["HeadDisp"] = df["Head"] + 1

        for (tbl, hdisp), df_g in df.groupby(["TableLabel", "HeadDisp"]):
            z_key = f"Z{tbl}{int(hdisp)}"
            z_keys_set.add(z_key)
            if selected_z and z_key not in selected_z:
                continue

            x = make_x_coordinate(df_g, start_pos, end_pos, index_zero_based=index_zero_based)
            y = df_g["Offset"].astype(float)

            all_entries.append(
                dict(
                    z_key=z_key,
                    x=x.values,
                    y=y.values,
                    legend=b.name,                 # ← 凡例=ファイル名
                    color=fname_color[b.name],     # ← このファイルの固定色
                    hovertext=[f"{b.name}<br>Z={z_key}<br>Index={int(ix)}<br>x={xx:.4f}<br>Offset={yy:.2f}%"
                               for ix, xx, yy in zip(df_g["Index"], x, y)]
                )
            )

    # 対象 Z 配列
    if selected_z:
        z_keys = [z for z in selected_z if z in z_keys_set]
    else:
        def z_key_sort(s: str):
            tbl = s[1] if len(s) >= 2 else "A"
            nums = re.findall(r"\d+", s)
            h = int(nums[0]) if nums else 0
            return (tbl, h)
        z_keys = sorted(z_keys_set, key=z_key_sort)

    n = len(z_keys)
    cols = 4
    rows = (n + cols - 1) // cols if n > 0 else 1
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=z_keys if z_keys else ["No Data"],
        horizontal_spacing=0.04, vertical_spacing=0.08
    )

    # Z → (row, col)
    z_to_rc: Dict[str, Tuple[int, int]] = {}
    for i, z in enumerate(z_keys):
        r = i // cols + 1
        c = i % cols + 1
        z_to_rc[z] = (r, c)

    # 凡例はファイル名ごとに1回
    added_legends = set()
    for ent in all_entries:
        z = ent["z_key"]
        if z not in z_to_rc:
            continue
        r, c = z_to_rc[z]
        label = ent["legend"]
        show_legend = False
        if label not in added_legends:
            show_legend = True
            added_legends.add(label)

        fig.add_trace(
            go.Scatter(
                x=ent["x"],
                y=ent["y"],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                hoverinfo="text",
                text=ent["hovertext"],
                line=dict(color=ent["color"], width=2),
                marker=dict(color=ent["color"], size=6)
            ),
            row=r, col=c
        )

    # 体裁
    for i, z in enumerate(z_keys):
        r = i // cols + 1
        c = i % cols + 1
        fig.update_xaxes(title_text="位置（StartPos→EndPos）", row=r, col=c)
        fig.update_yaxes(title_text="Offset [%]", row=r, col=c, zeroline=True)

    fig.update_layout(
        height=max(400, 350 * rows),
        title="Z{MntTable}{Head} ごとの Offset プロット（凡例=ファイル名／色固定／選択フィルタ）",
        legend_title="ファイル名",
        template="plotly_white"
    )
    return fig


# -----------------------------
# サイドバー UI
# -----------------------------
st.sidebar.title("設定")
index_zero_based = st.sidebar.checkbox(
    "Indexは0始まり（0,1,2,...）として横軸に線形割付する", value=True
)

# -----------------------------
# メイン UI
# -----------------------------
st.title("EctForce 形式プロッタ（ファイル名で色分け）")
# st.caption("225行=Start/End, 245行=ヘッダー（MntTable,Head,Index,Offset）。複数ファイルをアップロードし、Z{A/B}{Head+1} を選択して 4 列サブプロットで重ね描き（ファイル名で色固定）。")

uploaded_files = st.file_uploader(
    "EctForce 互換の *.sts を複数選択（凡例=ファイル名／色固定）",
    type=["sts"],
    accept_multiple_files=True
)

if uploaded_files:
    bundles: List[FileBundle] = []
    texts_for_scan: List[str] = []
    for uf in uploaded_files:
        try:
            lines = uf.getvalue().decode("utf-8", errors="ignore").splitlines()
            bundles.append(FileBundle(name=uf.name, lines=lines))
            texts_for_scan.append("\n".join(lines))
        except Exception as e:
            st.error(f"{uf.name} の読込に失敗: {e}")

    # Z 候補（キャッシュ）
    try:
        z_choices = scan_z_keys_cached(texts_for_scan)
    except Exception as e:
        st.error(f"Z 候補の抽出に失敗しました: {e}")
        z_choices = []

    # col1, col2 = st.columns([2, 1])
    # with col1:
    z_selected = st.multiselect(
        "表示する Z{MntTable}{Head}（未選択=全表示）",
        options=z_choices,
        default=[]
    )
    try:
        fig = build_figure(
            bundles=bundles,
            index_zero_based=index_zero_based,
            selected_z=z_selected if z_selected else None
        )
        st.plotly_chart(fig, use_container_width=True)

        # ダウンロード（HTML）
        html_bytes = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="図をHTMLでダウンロード",
            data=html_bytes,
            file_name="cogging_map.html",
            mime="text/html"
        )
    except Exception as e:
        st.error(f"描画に失敗しました: {e}")

    # with col2:
    #     st.markdown("#### ヒント")
    #     st.markdown("- 同一ファイル名は **全サブプロットで同じ色**になります。")
    #     st.markdown("- 色数が足りない場合は **パレットを繰り返し**適用します。")
    #     st.markdown("- 必要ならカラーパレットを UI から選択できるように拡張できます。")
else:
    st.info("EctForce 互換のテキストファイルをアップロードしてください。")
