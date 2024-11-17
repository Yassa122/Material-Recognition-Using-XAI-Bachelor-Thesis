"use client"; // Mark this as a Client Component

import dynamic from "next/dynamic";

const PlotlyChart = dynamic(() => import("react-plotly.js"), { ssr: false });

const ShapExplanationChart = () => {
  // Sample features for SHAP plot
  const features = [
    "MKT SIGMA",
    "TURN",
    "ROA",
    "MB",
    "SPECIFIC RET",
    "LOGSIZE",
    "SLB",
    "SIGMA",
    "CASHETR",
    "NCSEW",
    "LEV",
    "OPAQUE",
    "CEODWER",
  ];

  // Simulated SHAP values for each feature
  const shapValues = features.map(() =>
    Array.from({ length: 100 }, () => Math.random() * 2 - 1)
  );

  // Custom colors for markers
  const customColors = [
    "#FF6B6B", // Red
    "#1FAB89", // Green
    "#FFD700", // Gold
    "#6495ED", // Cornflower Blue
    "#FF69B4", // Hot Pink
    "#8A2BE2", // Blue Violet
    "#FFA07A", // Light Salmon
    "#20B2AA", // Light Sea Green
    "#00CED1", // Dark Turquoise
    "#FF4500", // Orange Red
    "#9370DB", // Medium Purple
    "#3CB371", // Medium Sea Green
    "#F08080", // Light Coral
  ];

  return (
    <div className="bg-sidebarBg p-6 rounded-xl shadow-lg">
      <div className="bg-zinc-900 p-6 rounded-xl shadow-lg">
        <h3 className="text-gray-200 text-lg font-bold mb-4">
          SHAP Explanation Chart
        </h3>
        <PlotlyChart
          data={shapValues.map((values, index) => ({
            type: "violin",
            x: values,
            y: Array(values.length).fill(features[index]), // Align y-axis categories
            points: "all",
            box: { visible: false },
            meanline: { visible: true },
            marker: {
              color: values.map(
                (val, i) => customColors[index % customColors.length]
              ), // Cycle through custom colors
              opacity: 0.8,
            },
            line: { color: customColors[index % customColors.length] },
            hoverinfo: "x+y",
            orientation: "h",
          }))}
          layout={{
            title: {
              text: "<b>SHAP Values (Impact on Model Output)</b>",
              font: { color: "white", size: 16 },
            },
            yaxis: {
              title: "",
              automargin: true,
              tickfont: { color: "#A0AEC0", size: 12 },
              tickcolor: "#A0AEC0",
              categoryorder: "total ascending", // Optional: Order by SHAP magnitude
            },
            xaxis: {
              title: {
                text: "<b>SHAP Value</b>",
                font: { color: "#A0AEC0", size: 14 },
              },
              tickfont: { color: "#A0AEC0", size: 12 },
              tickcolor: "#A0AEC0",
            },
            plot_bgcolor: "rgba(0, 0, 0, 0)",
            paper_bgcolor: "rgba(0, 0, 0, 0)",
            margin: { l: 120, r: 50, t: 50, b: 40 },
            showlegend: false,
            hoverlabel: {
              bgcolor: "#2D3748",
              font: { color: "white", size: 12 },
            },
          }}
          style={{ width: "100%", height: "450px" }}
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  );
};

export default ShapExplanationChart;
