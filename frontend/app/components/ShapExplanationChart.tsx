"use client"; // Mark this as a Client Component

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import Plotly from "plotly.js-basic-dist"; // Import Plotly directly
import axios from "axios"; // For making HTTP requests
import { FaUpload, FaDownload } from "react-icons/fa"; // Optional: Icons for buttons

// Dynamically import react-plotly.js to prevent SSR issues
const PlotlyChart = dynamic(() => import("react-plotly.js"), { ssr: false });

const ShapExplanationChart = () => {
  const [status, setStatus] = useState("idle"); // "idle" | "running" | "completed" | "error"
  const [message, setMessage] = useState("");
  const [shapValues, setShapValues] = useState([]);
  const [features, setFeatures] = useState([]);
  const [errorDetail, setErrorDetail] = useState("");
  const [uploading, setUploading] = useState(false); // Indicates if a file is being uploaded
  const [selectedFile, setSelectedFile] = useState(null); // Stores the selected file

  const fileInputRef = useRef(null); // Reference to the hidden file input
  const plotRef = useRef(null); // Reference to the Plotly chart

  const fetchShapStatus = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/shap_status");
      const data = response.data;

      setStatus(data.status);
      setMessage(data.message);

      if (data.status === "completed") {
        const { shap_values, features: featureNames } = data.result;

        // Sort features by the average magnitude of their SHAP values and select the top 5
        const averagedShapValues = shap_values.map(
          (values) =>
            values.reduce((sum, val) => sum + Math.abs(val), 0) / values.length
        );
        const sortedIndices = averagedShapValues
          .map((avg, index) => ({ avg, index }))
          .sort((a, b) => b.avg - a.avg)
          .slice(0, 5) // Limit to top 5 features
          .map((item) => item.index);

        const filteredShapValues = sortedIndices.map(
          (index) => shap_values[index]
        );
        const filteredFeatures = sortedIndices.map(
          (index) => featureNames[index]
        );

        setShapValues(filteredShapValues);
        setFeatures(filteredFeatures);
      } else if (data.status === "error") {
        setErrorDetail(
          data.error_detail || "An error occurred during SHAP explanation."
        );
      }
    } catch (error) {
      console.error("Error fetching SHAP status:", error);
      setStatus("error");
      setMessage("Failed to fetch SHAP status.");
      setErrorDetail(error.message || "Unknown error.");
    }
  };

  useEffect(() => {
    fetchShapStatus();
    const interval = setInterval(() => {
      fetchShapStatus();
    }, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const uploadShapFile = async (file) => {
    setUploading(true);
    setErrorDetail("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        "http://127.0.0.1:5000/start_shap_explanation",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.status === 202) {
        setStatus("running");
        setMessage("SHAP explanation started. Processing...");
      } else {
        setStatus("error");
        setMessage("Unexpected response from the server.");
      }
    } catch (error) {
      console.error("Error starting SHAP explanation:", error);
      setStatus("error");
      setMessage("Failed to start SHAP explanation.");
      setErrorDetail(error.response?.data?.error || error.message);
    } finally {
      setUploading(false);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset the file input
      }
    }
  };

  const generatePlotlyData = () => {
    if (
      !shapValues ||
      shapValues.length === 0 ||
      !features ||
      features.length === 0
    ) {
      return [];
    }

    const customColors = [
      "#FF6B6B", // Red
      "#1FAB89", // Green
      "#FFD700", // Gold
      "#6495ED", // Cornflower Blue
      "#FF69B4", // Hot Pink
    ];

    return shapValues.map((values, index) => ({
      type: "violin",
      x: values,
      y: Array(values.length).fill(features[index]), // Align y-axis categories
      points: "all",
      box: { visible: false },
      meanline: { visible: true },
      marker: {
        color: customColors[index % customColors.length],
        opacity: 0.8,
      },
      line: { color: customColors[index % customColors.length] },
      hoverinfo: "x+y",
      orientation: "h",
    }));
  };

  const downloadChart = async () => {
    if (plotRef.current) {
      try {
        await Plotly.downloadImage(plotRef.current, {
          format: "png",
          filename: "shap_explanation_chart",
          width: 800,
          height: 450,
        });
      } catch (error) {
        console.error("Error downloading chart:", error);
      }
    }
  };

  return (
    <div className="bg-sidebarBg p-6 rounded-xl shadow-lg">
      <div className="bg-zinc-900 p-6 rounded-xl shadow-lg">
        <h3 className="text-gray-200 text-lg font-bold mb-4">
          SHAP Explanation Chart
        </h3>

        {status === "idle" && (
          <div className="mb-6">
            <button
              onClick={() => fileInputRef.current.click()}
              className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={uploading}
            >
              <FaUpload className="mr-2" /> Start SHAP Explanation
            </button>
            <input
              type="file"
              accept=".csv"
              ref={fileInputRef}
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) setSelectedFile(file);
                uploadShapFile(file);
              }}
              className="hidden"
            />
          </div>
        )}

        {status === "completed" &&
          shapValues.length > 0 &&
          features.length > 0 && (
            <>
              <div ref={plotRef}>
                <PlotlyChart
                  data={generatePlotlyData()}
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
                    margin: { l: 120, r: 50, t: 50, b: 40 },
                    showlegend: false,
                  }}
                  style={{ width: "100%", height: "450px" }}
                  config={{ displayModeBar: false }}
                />
              </div>
              <div className="mt-4">
                <button
                  onClick={downloadChart}
                  className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md shadow focus:outline-none focus:ring-2 focus:ring-green-500"
                >
                  <FaDownload className="mr-2" /> Download Chart
                </button>
              </div>
            </>
          )}
      </div>
    </div>
  );
};

export default ShapExplanationChart;
