"use client"; // Mark this as a Client Component

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import axios from "axios"; // For making HTTP requests
import { FaUpload } from "react-icons/fa"; // Optional: Icon for the upload button

// Dynamically import react-plotly.js to prevent SSR issues
const PlotlyChart = dynamic(() => import("react-plotly.js"), { ssr: false });

const ShapExplanationChart = () => {
  const [status, setStatus] = useState("idle"); // "idle" | "running" | "completed" | "error"
  const [message, setMessage] = useState("");
  const [shapValues, setShapValues] = useState([]);
  const [features, setFeatures] = useState([]);
  const [plotUrl, setPlotUrl] = useState("");
  const [errorDetail, setErrorDetail] = useState("");
  const [uploading, setUploading] = useState(false); // Indicates if a file is being uploaded
  const [selectedFile, setSelectedFile] = useState(null); // Stores the selected file

  const fileInputRef = useRef(null); // Reference to the hidden file input

  const fetchShapStatus = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/shap_status");
      const data = response.data;

      setStatus(data.status);
      setMessage(data.message);

      if (data.status === "completed") {
        const {
          shap_values,
          features: featureNames,
          plot_filename,
        } = data.result;

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
        setPlotUrl(`http://127.0.0.1:5000/download_shap_plot/${plot_filename}`); // URL to download the SHAP summary plot image
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

  // Handle the click on the "Start SHAP Explanation" button
  const handleStartShap = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click(); // Trigger the hidden file input
    }
  };

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      uploadShapFile(file);
    }
  };

  // Upload the selected file to the Flask backend
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

  return (
    <div className="bg-sidebarBg p-6 rounded-xl shadow-lg">
      <div className="bg-zinc-900 p-6 rounded-xl shadow-lg">
        <h3 className="text-gray-200 text-lg font-bold mb-4">
          SHAP Explanation Chart
        </h3>

        {/* Button to start SHAP explanation */}
        {status === "idle" && (
          <div className="mb-6">
            <button
              onClick={handleStartShap}
              className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={uploading}
            >
              <FaUpload className="mr-2" /> Start SHAP Explanation
            </button>
            {/* Hidden file input */}
            <input
              type="file"
              accept=".csv"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        )}

        {/* Status Messages */}
        {status === "idle" && (
          <p className="text-gray-400">SHAP explanation has not started yet.</p>
        )}

        {status === "running" && (
          <div>
            <p className="text-gray-400 mb-4">{message}</p>
            <div className="flex items-center">
              <div className="w-full bg-gray-700 rounded-full h-4">
                <div
                  className="bg-blue-500 h-4 rounded-full"
                  style={{ width: "50%" }}
                ></div>
              </div>
              <span className="ml-2 text-gray-400">Processing...</span>
            </div>
          </div>
        )}

        {status === "error" && (
          <div>
            <p className="text-red-500 mb-4">Error: {message}</p>
            <p className="text-red-400">{errorDetail}</p>
          </div>
        )}

        {status === "completed" &&
          shapValues.length > 0 &&
          features.length > 0 && (
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
                  categoryorder: "total ascending",
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
          )}

        {/* SHAP Summary Plot Image */}
        {status === "completed" && plotUrl && (
          <div className="mt-6">
            <h4 className="text-gray-200 text-md font-semibold mb-2">
              SHAP Summary Plot Image
            </h4>
            <a
              href={plotUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline"
            >
              Download SHAP Summary Plot
            </a>
            <div className="mt-4">
              {/* <img
                src={plotUrl}
                alt="SHAP Summary Plot"
                className="w-full h-auto rounded-md shadow-lg"
              /> */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShapExplanationChart;
