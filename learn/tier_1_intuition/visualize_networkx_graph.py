"""
Tier 1 Intuition: NetworkX Graph Representation and Wrist Neighborhood Visualization.

Loads 21 hand keypoints from JSON, builds anatomical bone graph,
computes 1-hop and 2-hop neighborhoods of Node 0 (Wrist),
renders a high-contrast 2D Matplotlib figure, and exports topology to JSON.
"""

import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# 20 standard hand anatomical bone connections
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

LANDMARK_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
]

def load_keypoints(json_path: str):
    """Load coordinates from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "landmarks" in data:
        coords = {item["id"]: (item["x"], -item["y"]) for item in data["landmarks"]}
    else:
        flat = data.get("flat_coordinates", data)
        coords = {i: (flat[i * 3], -flat[i * 3 + 1]) for i in range(len(flat) // 3)}
    return coords

def build_skeleton_graph(coords: dict) -> nx.Graph:
    """Build NetworkX graph with 21 nodes and 20 skeletal bone edges."""
    G = nx.Graph()
    for i in range(21):
        x, y = coords[i]
        G.add_node(i, name=LANDMARK_NAMES[i], pos=(x, y))
    for u, v in HAND_BONES:
        G.add_edge(u, v)
    return G

def get_wrist_neighborhoods(G: nx.Graph, source_node: int = 0):
    """Identify 1-hop and 2-hop neighborhoods of the source node."""
    # 1-hop: immediate neighbors
    one_hop = set(G.neighbors(source_node))
    
    # 2-hop: neighbors of 1-hop excluding source and 1-hop
    two_hop = set()
    for n in one_hop:
        for nn in G.neighbors(n):
            if nn != source_node and nn not in one_hop:
                two_hop.add(nn)
                
    remaining = set(G.nodes()) - {source_node} - one_hop - two_hop
    return one_hop, two_hop, remaining

def render_visualization(G: nx.Graph, coords: dict, one_hop: set, two_hop: set, remaining: set, out_path: str):
    """Render high-contrast 2D graph visualization."""
    fig, ax = plt.subplots(figsize=(8, 9), dpi=150)
    fig.patch.set_facecolor("#fafaf9")
    ax.set_facecolor("#fafaf9")
    
    pos = {node: coords[node] for node in G.nodes()}
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#78716c", width=2.0, alpha=0.85)
    
    # Node colors adhering to WCAG AA high contrast
    node_colors = {}
    node_colors[0] = "#991b1b"  # Deep red for Wrist (Node 0)
    for n in one_hop:
        node_colors[n] = "#1e40af" # Deep blue for 1-hop
    for n in two_hop:
        node_colors[n] = "#b45309" # Amber for 2-hop
    for n in remaining:
        node_colors[n] = "#475569" # Slate for remaining
        
    color_list = [node_colors[n] for n in G.nodes()]
    node_sizes = [500 if n == 0 else (400 if n in one_hop or n in two_hop else 300) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_list, node_size=node_sizes, edgecolors="#1c1917", linewidths=1.5)
    
    # Node labels
    labels = {n: f"{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight="bold", font_color="#ffffff")
    
    # Off-node textual annotations for key landmarks
    for n in [0] + sorted(list(one_hop)):
        x, y = pos[n]
        offset_x = 0.03 if x >= 0.5 else -0.03
        ha = "left" if x >= 0.5 else "right"
        ax.text(x + offset_x, y, f"{LANDMARK_NAMES[n]} ({n})", fontsize=8, color="#1c1917",
                ha=ha, va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#d6d3d1", lw=1))
        
    ax.set_title("Hand Skeleton Graph: Wrist (Node 0) 1-Hop & 2-Hop Neighborhoods", fontsize=12, fontweight="bold", pad=15, color="#1c1917")
    
    # Custom legend
    custom_lines = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#991b1b', markersize=10, label='Wrist (Node 0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e40af', markersize=9, label=f'1-Hop: {sorted(list(one_hop))}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b45309', markersize=9, label=f'2-Hop: {sorted(list(two_hop))}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#475569', markersize=8, label=f'3-Hop+ ({len(remaining)} nodes)')
    ]
    ax.legend(handles=custom_lines, loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e7e5e4", fontsize=8.5)
    ax.axis("off")
    
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

def export_graph_data(G: nx.Graph, coords: dict, one_hop: set, two_hop: set, remaining: set, out_path: str):
    """Save graph metadata to JSON for web SVG canvas."""
    data = {
        "nodes": [
            {
                "id": n,
                "name": LANDMARK_NAMES[n],
                "x": round(coords[n][0], 4),
                "y": round(-coords[n][1], 4),
                "group": "source" if n == 0 else ("one_hop" if n in one_hop else ("two_hop" if n in two_hop else "other")),
                "hop_distance": 0 if n == 0 else (1 if n in one_hop else (2 if n in two_hop else 3))
            }
            for n in G.nodes()
        ],
        "edges": [{"source": u, "target": v} for u, v in G.edges()],
        "neighborhoods": {
            "source": 0,
            "one_hop": sorted(list(one_hop)),
            "two_hop": sorted(list(two_hop)),
            "remaining": sorted(list(remaining))
        }
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand skeleton graph neighborhoods")
    parser.add_argument("--json", type=str, default=None, help="Path to keypoints JSON")
    parser.add_argument("--out", type=str, default=None, help="Output image file")
    args = parser.parse_args()
    
    keypoint_file = args.json or str(Path(__file__).resolve().parent / "extracted_keypoints.json")
    if not Path(keypoint_file).exists():
        # Run extraction first to generate keypoint file
        from extract_mediapipe_keypoints import extract_keypoints, save_flat_json
        coords_raw = extract_keypoints()
        save_flat_json(coords_raw, keypoint_file)
        
    coords = load_keypoints(keypoint_file)
    G = build_skeleton_graph(coords)
    one_hop, two_hop, remaining = get_wrist_neighborhoods(G, 0)
    
    out_img = args.out or str(Path(__file__).resolve().parent / "wrist_neighborhood_graph.png")
    render_visualization(G, coords, one_hop, two_hop, remaining, out_img)
    
    # Also save image and json to web directory
    web_img = Path(__file__).resolve().parent.parent / "web" / "data" / "wrist_neighborhood_graph.png"
    render_visualization(G, coords, one_hop, two_hop, remaining, str(web_img))
    web_json = Path(__file__).resolve().parent.parent / "web" / "data" / "hand_graph_topology.json"
    export_graph_data(G, coords, one_hop, two_hop, remaining, str(web_json))
    
    print("=== TIER 1: NETWORKX GRAPH VISUALIZATION ===")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Node 0 (Wrist) 1-hop neighbors: {sorted(list(one_hop))}")
    print(f"Node 0 (Wrist) 2-hop neighbors: {sorted(list(two_hop))}")
    print(f"Remaining nodes (>2 hops): {sorted(list(remaining))}")
    print(f"Saved visualization plot to: {out_img}")
