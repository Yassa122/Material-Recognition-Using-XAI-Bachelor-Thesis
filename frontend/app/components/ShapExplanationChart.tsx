// components/ShapExplanationChart.tsx
import Plot from "react-plotly.js";

const ShapExplanationChart = () => {
  // Sample data for SHAP plot
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

  // Generate simulated SHAP values for each feature
  const shapValues = Array.from({ length: features.length }, () =>
    Array.from({ length: 100 }, () => Math.random() * 2 - 1)
  );

  return (
    <div className="bg-sidebarBg p-6 rounded-xl shadow-lg">
      <h3 className="text-gray-200 text-lg font-bold mb-4">
        SHAP Explanation Chart
      </h3>
      <Plot
        data={[
          {
            type: "violin",
            y: features,
            x: shapValues.flat(),
            points: "all",
            box: { visible: false },
            meanline: { visible: true },
            marker: {
              color: shapValues
                .flat()
                .map((val) => (val > 0 ? "#FF6B6B" : "#1FAB89")),
              opacity: 0.6,
            },
            line: { color: "white" },
            colorscale: "RdBu", // Red-Blue color scale for better impact visibility
            hoverinfo: "x+y",
            orientation: "h",
          },
        ]}
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
          margin: { l: 100, r: 50, t: 50, b: 40 },
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
  );
};

export default ShapExplanationChart;
