import { useRef, useEffect } from "react";
import * as d3 from "d3";

export default function NetworkGraph({ molecule = {} }) {
    const ref = useRef();

    useEffect(() => {
        const atoms = molecule?.atoms || [];
        const bonds = molecule?.bonds || [];

        // Handle invalid data
        if (atoms.length === 0 || bonds.length === 0) {
            console.warn("Invalid molecule data: missing atoms or bonds", molecule);
            d3.select(ref.current).selectAll("*").remove(); // Clear previous content
            d3.select(ref.current)
                .append("text")
                .attr("x", "50%")
                .attr("y", "50%")
                .attr("text-anchor", "middle")
                .attr("fill", "red")
                .text("Invalid molecule data: missing atoms or bonds.");
            return;
        }

        // Clear previous content
        d3.select(ref.current).selectAll("*").remove();

        // Dimensions
        const container = ref.current.parentNode;
        const width = container.offsetWidth || 600; // Container width
        const height = container.offsetHeight || 600; // Container height

        // SVG setup
        const svg = d3.select(ref.current)
            .attr("width", width)
            .attr("height", height);

        // Define color scale for atom types
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
            .domain([...new Set(atoms.map((atom) => atom.element))]);

        // Prepare nodes and links
        const nodes = atoms.map((atom, i) => ({
            id: i,
            name: atom.element || "Unknown",
            element: atom.element,
        }));

        const links = bonds.map((bond) => ({
            source: bond.startAtomIndex,
            target: bond.endAtomIndex,
        }));

        // Calculate scaling factor and offsets to center the graph
        const scaleFactor = Math.min(width / 600, height / 600); // Scale to fit the container
        const offsetX = (width - 600 * scaleFactor) / 2; // Center horizontally
        const offsetY = (height - 600 * scaleFactor) / 2; // Center vertically

        // Apply transformations
        const graphGroup = svg.append("g")
            .attr("transform", `translate(${offsetX}, ${offsetY}) scale(${scaleFactor})`);

        // Force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id((d) => d.id).distance(50))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(300, 300)) // Center relative to the original 600x600 size
            .on("tick", ticked);

        // Draw links (edges)
        const link = graphGroup.append("g")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke-width", 1.5);

        // Draw nodes (atoms)
        const node = graphGroup.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", 10)
            .attr("fill", (d) => colorScale(d.element))
            .call(drag(simulation));

        // Add labels for nodes
        const label = graphGroup.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("font-size", "10px")
            .attr("fill", "#fff")
            .attr("text-anchor", "middle")
            .text((d) => d.name);

        // Tick function to update positions
        function ticked() {
            link
                .attr("x1", (d) => d.source.x)
                .attr("y1", (d) => d.source.y)
                .attr("x2", (d) => d.target.x)
                .attr("y2", (d) => d.target.y);

            node
                .attr("cx", (d) => d.x)
                .attr("cy", (d) => d.y);

            label
                .attr("x", (d) => d.x)
                .attr("y", (d) => d.y - 15); // Adjust label position slightly above node
        }

        // Drag handlers
        function drag(simulation) {
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }
    }, [molecule]);

    return <svg ref={ref} style={{ width: "100%", height: "100%" }}></svg>;
}
