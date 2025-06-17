#!/usr/bin/env python3
"""
Codebase Graph Visualizer
Generates an interactive graph visualization of codebase structure.
Run from the root directory of your project.
"""

import os
import json
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime


class CodebaseGraphGenerator:
    def __init__(self, max_depth=None, show_hidden=False, include_files=True):
        self.max_depth = max_depth
        self.show_hidden = show_hidden
        self.include_files = include_files
        self.root_path = None
        
        # Common directories and files to ignore
        self.ignore_dirs = {
            '.git', '.svn', '.hg',  # Version control
            'node_modules', 'bower_components',  # JavaScript
            '__pycache__', '.pytest_cache', 'venv', 'env', '.env',  # Python
            'target', 'build', 'dist', 'out',  # Build outputs
            '.idea', '.vscode', '.vs',  # IDEs
            'vendor',  # Dependencies
            'coverage', '.coverage',  # Coverage reports
            'logs', 'log',  # Log files
            '.next', '.nuxt',  # Framework specific
            'bin', 'obj',  # .NET
        }
        
        # File extension categories for color coding
        self.file_categories = {
            'code': {
                'extensions': {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
                             '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
                             '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte'},
                'color': '#4CAF50',  # Green
                'icon': '💻'
            },
            'config': {
                'extensions': {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
                             '.xml', '.env', '.properties', '.gitignore', '.dockerignore'},
                'color': '#FF9800',  # Orange
                'icon': '⚙️'
            },
            'docs': {
                'extensions': {'.md', '.txt', '.rst', '.adoc', '.tex', '.pdf'},
                'color': '#2196F3',  # Blue
                'icon': '📄'
            },
            'media': {
                'extensions': {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
                             '.mp4', '.avi', '.mov', '.mp3', '.wav'},
                'color': '#E91E63',  # Pink
                'icon': '🎨'
            },
            'data': {
                'extensions': {'.csv', '.tsv', '.xlsx', '.xls', '.db', '.sqlite', '.sql'},
                'color': '#9C27B0',  # Purple
                'icon': '📊'
            }
        }

    def should_ignore_dir(self, dir_name):
        """Check if directory should be ignored."""
        if not self.show_hidden and dir_name.startswith('.'):
            return True
        return dir_name in self.ignore_dirs

    def should_ignore_file(self, file_name):
        """Check if file should be ignored."""
        if not self.show_hidden and file_name.startswith('.'):
            return True
        return False

    def get_file_category(self, file_path):
        """Categorize file by extension."""
        ext = file_path.suffix.lower()
        for category, info in self.file_categories.items():
            if ext in info['extensions']:
                return category
        return 'other'

    def get_relative_path(self, path):
        """Get relative path from root."""
        try:
            return str(path.relative_to(self.root_path))
        except ValueError:
            return str(path)

    def analyze_structure(self, path, current_depth=0, parent_id=None):
        """Recursively analyze directory structure and build graph data."""
        nodes = []
        links = []
        
        if self.max_depth and current_depth >= self.max_depth:
            return nodes, links
        
        try:
            items = list(path.iterdir())
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                if item.is_dir():
                    if self.should_ignore_dir(item.name):
                        continue
                    
                    # Count contents for size representation
                    try:
                        contents = list(item.iterdir())
                        file_count = sum(1 for p in contents if p.is_file() and not self.should_ignore_file(p.name))
                        dir_count = sum(1 for p in contents if p.is_dir() and not self.should_ignore_dir(p.name))
                        total_size = file_count + dir_count
                    except PermissionError:
                        file_count = dir_count = total_size = 0
                    
                    node_id = self.get_relative_path(item)
                    
                    # Create directory node
                    nodes.append({
                        'id': node_id,
                        'name': item.name,
                        'type': 'directory',
                        'relativePath': node_id,
                        'size': max(10, min(50, total_size * 2)),  # Scale size for visualization
                        'color': '#607D8B',  # Blue Grey for directories
                        'icon': '📁',
                        'fileCount': file_count,
                        'dirCount': dir_count
                    })
                    
                    # Create link to parent
                    if parent_id is not None:
                        links.append({
                            'source': parent_id,
                            'target': node_id,
                            'type': 'directory'
                        })
                    
                    # Recursively process subdirectory
                    sub_nodes, sub_links = self.analyze_structure(
                        item, current_depth + 1, node_id
                    )
                    nodes.extend(sub_nodes)
                    links.extend(sub_links)
                
                elif self.include_files and item.is_file():
                    if self.should_ignore_file(item.name):
                        continue
                    
                    try:
                        size = item.stat().st_size
                    except (OSError, PermissionError):
                        size = 0
                    
                    category = self.get_file_category(item)
                    category_info = self.file_categories.get(category, {
                        'color': '#757575',
                        'icon': '📎'
                    })
                    
                    node_id = self.get_relative_path(item)
                    
                    # Create file node
                    nodes.append({
                        'id': node_id,
                        'name': item.name,
                        'type': 'file',
                        'relativePath': node_id,
                        'size': max(5, min(20, size / 1024)),  # Scale size (KB)
                        'color': category_info['color'],
                        'icon': category_info['icon'],
                        'category': category,
                        'fileSize': size
                    })
                    
                    # Create link to parent
                    if parent_id is not None:
                        links.append({
                            'source': parent_id,
                            'target': node_id,
                            'type': 'file'
                        })
        
        except PermissionError:
            pass
        
        return nodes, links

    def generate_html(self, nodes, links, root_name):
        """Generate HTML file with D3.js graph visualization."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codebase Structure Graph - {root_name}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: white;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        button {{
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        button:hover {{
            background: #45a049;
        }}
        
        input[type="range"] {{
            width: 100px;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 14px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        
        #graph {{
            border: 1px solid #444;
            border-radius: 8px;
            background: #2a2a2a;
        }}
        
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            font-size: 12px;
            z-index: 1000;
            max-width: 300px;
        }}
        
        .stats {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-card {{
            background: #333;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Codebase Structure Graph</h1>
            <h2>📂 {root_name}</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>Zoom:</label>
                <button onclick="zoomIn()">🔍 +</button>
                <button onclick="zoomOut()">🔍 -</button>
                <button onclick="resetZoom()">↺ Reset</button>
            </div>
            <div class="control-group">
                <label>Layout:</label>
                <button onclick="restartSimulation()">🔄 Restart</button>
                <button onclick="centerGraph()">📍 Center</button>
            </div>
            <div class="control-group">
                <label>Force Strength:</label>
                <input type="range" id="forceSlider" min="10" max="1000" value="300" 
                       onchange="updateForces(this.value)">
                <span id="forceValue">300</span>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #607D8B;"></div>
                <span>📁 Directories</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #4CAF50;"></div>
                <span>💻 Code Files</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF9800;"></div>
                <span>⚙️ Config Files</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2196F3;"></div>
                <span>📄 Documentation</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #E91E63;"></div>
                <span>🎨 Media Files</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9C27B0;"></div>
                <span>📊 Data Files</span>
            </div>
        </div>
        
        <div id="graph"></div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number" id="nodeCount">{len(nodes)}</div>
                <div>Total Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="linkCount">{len(links)}</div>
                <div>Connections</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="dirCount">{len([n for n in nodes if n['type'] == 'directory'])}</div>
                <div>Directories</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="fileCount">{len([n for n in nodes if n['type'] == 'file'])}</div>
                <div>Files</div>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const graphData = {{
            nodes: {json.dumps(nodes, indent=2)},
            links: {json.dumps(links, indent=2)}
        }};
        
        // Graph dimensions
        const width = window.innerWidth - 40;
        const height = 800;
        
        // Create SVG
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create container for zoom
        const container = svg.append("g");
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", function(event) {{
                container.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Create tooltip
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
        
        // Force simulation
        let simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 2));
        
        // Create links
        const link = container.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("stroke", "#666")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => d.type === "directory" ? 2 : 1);
        
        // Create nodes
        const node = container.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .style("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", function(event, d) {{
                let content = `
                    <strong>${{d.icon}} ${{d.name}}</strong><br>
                    <strong>Path:</strong> ${{d.relativePath}}<br>
                    <strong>Type:</strong> ${{d.type}}
                `;
                
                if (d.type === "directory") {{
                    content += `<br><strong>Files:</strong> ${{d.fileCount}}<br><strong>Subdirs:</strong> ${{d.dirCount}}`;
                }} else {{
                    content += `<br><strong>Category:</strong> ${{d.category}}<br><strong>Size:</strong> ${{formatBytes(d.fileSize)}}`;
                }}
                
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(content)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }});
        
        // Add labels
        const label = container.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .text(d => d.name.length > 15 ? d.name.substring(0, 12) + "..." : d.name)
            .attr("font-size", "10px")
            .attr("dx", d => d.size + 3)
            .attr("dy", "0.35em")
            .attr("fill", "white")
            .style("pointer-events", "none");
        
        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Utility functions
        function formatBytes(bytes) {{
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }}
        
        // Control functions
        function zoomIn() {{
            svg.transition().call(zoom.scaleBy, 1.5);
        }}
        
        function zoomOut() {{
            svg.transition().call(zoom.scaleBy, 1 / 1.5);
        }}
        
        function resetZoom() {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }}
        
        function restartSimulation() {{
            simulation.alpha(1).restart();
        }}
        
        function centerGraph() {{
            const bounds = container.node().getBBox();
            const fullWidth = width;
            const fullHeight = height;
            const widthRatio = fullWidth / bounds.width;
            const heightRatio = fullHeight / bounds.height;
            const scale = 0.8 * Math.min(widthRatio, heightRatio);
            const translate = [fullWidth / 2 - scale * (bounds.x + bounds.width / 2), 
                             fullHeight / 2 - scale * (bounds.y + bounds.height / 2)];
            
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
        }}
        
        function updateForces(value) {{
            document.getElementById('forceValue').textContent = value;
            simulation.force("charge", d3.forceManyBody().strength(-value));
            simulation.alpha(1).restart();
        }}
        
        // Auto-center on load
        setTimeout(centerGraph, 1000);
    </script>
</body>
</html>
        """
        return html_template

    def format_size(self, size_bytes):
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def print_terminal_structure(self, nodes):
        """Print organized file structure in terminal."""
        print("\n" + "="*80)
        print("📂 CODEBASE STRUCTURE - ORGANIZED BY RELATIVE PATH")
        print("="*80)
        
        # Separate directories and files
        directories = [n for n in nodes if n['type'] == 'directory']
        files = [n for n in nodes if n['type'] == 'file']
        
        # Sort by relative path
        directories.sort(key=lambda x: x['relativePath'])
        files.sort(key=lambda x: x['relativePath'])
        
        # Print directories
        if directories:
            print(f"\n📁 DIRECTORIES ({len(directories)}):")
            print("-" * 50)
            for dir_node in directories:
                path = dir_node['relativePath']
                name = dir_node['name']
                file_count = dir_node.get('fileCount', 0)
                dir_count = dir_node.get('dirCount', 0)
                
                if path == '.':
                    print(f"🏠 {name}/ ({file_count} files, {dir_count} subdirs)")
                else:
                    depth = path.count(os.sep)
                    indent = "  " * depth
                    print(f"{indent}📁 {path}/ ({file_count} files, {dir_count} subdirs)")
        
        # Print files organized by category
        if files:
            # Group files by category
            files_by_category = {}
            for file_node in files:
                category = file_node.get('category', 'other')
                if category not in files_by_category:
                    files_by_category[category] = []
                files_by_category[category].append(file_node)
            
            # Category display order and info
            category_info = {
                'code': {'name': 'CODE FILES', 'icon': '💻'},
                'config': {'name': 'CONFIG FILES', 'icon': '⚙️'},
                'docs': {'name': 'DOCUMENTATION', 'icon': '📄'},
                'media': {'name': 'MEDIA FILES', 'icon': '🎨'},
                'data': {'name': 'DATA FILES', 'icon': '📊'},
                'other': {'name': 'OTHER FILES', 'icon': '📎'}
            }
            
            for category in ['code', 'config', 'docs', 'media', 'data', 'other']:
                if category in files_by_category:
                    category_files = files_by_category[category]
                    info = category_info[category]
                    
                    print(f"\n{info['icon']} {info['name']} ({len(category_files)}):")
                    print("-" * 50)
                    
                    for file_node in category_files:
                        path = file_node['relativePath']
                        file_size = file_node.get('fileSize', 0)
                        size_str = self.format_size(file_size)
                        
                        depth = path.count(os.sep)
                        indent = "  " * depth
                        print(f"{indent}{info['icon']} {path} ({size_str})")
        
        # Print summary statistics
        total_files = len(files)
        total_dirs = len(directories)
        total_size = sum(f.get('fileSize', 0) for f in files)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print("-" * 50)
        print(f"📁 Total Directories: {total_dirs}")
        print(f"📄 Total Files: {total_files}")
        print(f"💾 Total Size: {self.format_size(total_size)}")
        
        # Category breakdown
        if files:
            print(f"\n📈 FILES BY CATEGORY:")
            print("-" * 30)
            for category in ['code', 'config', 'docs', 'media', 'data', 'other']:
                if category in files_by_category:
                    count = len(files_by_category[category])
                    info = category_info[category]
                    percentage = (count / total_files) * 100
                    print(f"{info['icon']} {info['name'].title()}: {count} ({percentage:.1f}%)")

    def run(self, root_path="."):
        """Run the analysis and generate graph visualization."""
        self.root_path = Path(root_path).resolve()
        
        print(f"🌳 Analyzing codebase structure: {self.root_path}")
        print(f"📊 Generating interactive graph...")
        
        # Add root node
        root_node = {
            'id': '.',
            'name': self.root_path.name,
            'type': 'directory',
            'relativePath': '.',
            'size': 30,
            'color': '#FF5722',  # Deep Orange for root
            'icon': '🏠',
            'fileCount': 0,
            'dirCount': 0
        }
        
        # Analyze structure
        nodes, links = self.analyze_structure(self.root_path, parent_id='.')
        
        # Insert root node at the beginning
        nodes.insert(0, root_node)
        
        # Update root node counts
        root_node['fileCount'] = len([n for n in nodes if n['type'] == 'file'])
        root_node['dirCount'] = len([n for n in nodes if n['type'] == 'directory']) - 1
        
        # Print terminal structure
        self.print_terminal_structure(nodes)
        
        print(f"\n🔗 Created {len(links)} connections")
        
        # Generate HTML
        html_content = self.generate_html(nodes, links, self.root_path.name)
        
        # Save to file
        output_file = self.root_path / "codebase_graph.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ Graph saved to: {output_file}")
        print(f"🌐 Opening in browser...")
        
        # Open in browser
        webbrowser.open(f"file://{output_file.absolute()}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive graph visualization of codebase structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_structure.py                     # Analyze current directory
  python graph_structure.py --max-depth 4      # Limit to 4 levels deep
  python graph_structure.py --no-files         # Show only directories
  python graph_structure.py --show-hidden      # Include hidden files/dirs
  python graph_structure.py /path/to/project   # Analyze specific directory
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root directory to analyze (default: current directory)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Maximum depth to traverse (default: 5)'
    )
    
    parser.add_argument(
        '--no-files',
        action='store_true',
        help='Show only directories, not files'
    )
    
    parser.add_argument(
        '--show-hidden',
        action='store_true',
        help='Include hidden files and directories'
    )
    
    args = parser.parse_args()
    
    generator = CodebaseGraphGenerator(
        max_depth=args.max_depth,
        show_hidden=args.show_hidden,
        include_files=not args.no_files
    )
    
    try:
        generator.run(args.path)
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()