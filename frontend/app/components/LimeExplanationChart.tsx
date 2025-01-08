"use client";
import { useState, useEffect } from "react";
import axios from "axios";

// 1) Import and register chart.js modules
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// 2) Import Bar instead of HorizontalBar
import { Bar } from "react-chartjs-2";

const LimeExplanationChart = () => {
  const [limeData, setLimeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const LIME_API_URL =
    "http://localhost:5000/api/explain_biological_activity_lime";

  useEffect(() => {
    const fetchLimeData = async () => {
      try {
        const response = await axios.get(LIME_API_URL);
        setLimeData(response.data);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching LIME data:", err);
        setError(true);
        setLoading(false);
      }
    };
    fetchLimeData();
  }, []);

  if (loading) {
    return <div className="text-center">Loading LIME...</div>;
  }
  if (error || !limeData) {
    return (
      <div className="text-center text-red-500">Error loading LIME data.</div>
    );
  }

  // Assuming limeData has { explanation: [ [label, weight], ... ] }
  const explanationArray = limeData.explanation || [];
  const featureLabels = explanationArray.map((item) => item[0]);
  const weights = explanationArray.map((item) => item[1]);

  const data = {
    labels: featureLabels,
    datasets: [
      {
        label: "LIME Weights",
        data: weights,
        backgroundColor: "rgba(255, 99, 132, 0.6)",
      },
    ],
  };

  // 3) For a horizontal bar, set indexAxis = "y"
  const options = {
    indexAxis: "y",
    responsive: true,
    scales: {
      x: { beginAtZero: true },
    },
  };

  return (
    <div className="mt-4">
      <h2 className="text-xl font-semibold mb-2">LIME Explanation</h2>
      <Bar data={data} options={options} />
    </div>
  );
};

export default LimeExplanationChart;
