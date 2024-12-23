import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

const LimeCharts = () => {
  const [limeData, setLimeData] = useState([]);
  const [pieData, setPieData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch LIME explanation data
  useEffect(() => {
    const fetchLimeData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/lime_status");
        const data = await response.json();

        if (data.status === "completed") {
          // Extract weights and feature names from the result
          const weights = data.result.weights.map(([feature, weight]) => ({
            name: feature,
            value: Math.abs(weight), // Use absolute weights for comparison
          }));

          // Prepare Pie Data
          const pieData = data.result.weights.map(([feature, weight]) => ({
            name: feature,
            value: Math.abs(weight), // Use absolute weights
            color: generateRandomColor(), // Assign a random color to each feature
          }));

          setLimeData(weights);
          setPieData(pieData);
          setLoading(false);
        } else if (data.status === "error") {
          setError(data.message);
          setLoading(false);
        } else {
          setError("LIME explanation is not yet completed.");
          setLoading(false);
        }
      } catch (err) {
        setError("Failed to fetch LIME explanation.");
        setLoading(false);
      }
    };

    fetchLimeData();
  }, []);

  // Function to generate random colors for Pie Chart
  const generateRandomColor = () => {
    const letters = "0123456789ABCDEF";
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  };

  if (loading) return <p>Loading LIME Explanation...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div className="flex flex-col md:flex-row space-y-6 md:space-y-0 md:space-x-6 bg-sidebarBg p-6 rounded-lg shadow-lg">
      {/* Bar Chart */}
      <div className="w-full md:w-1/2 bg-[#202020] p-4 rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-gray-300 font-semibold">LIME Feature Weights</h3>
          <span className="text-green-400 font-semibold text-sm">+15%</span>
        </div>
        <p className="text-3xl font-bold text-white mb-1">Feature Importance</p>
        <p className="text-gray-400 text-sm mb-6">Based on LIME Explanation</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart className="text-black" data={limeData}>
            <XAxis
              dataKey="name"
              tick={{ fill: "#A0AEC0" }}
              interval={0}
              angle={-45}
              textAnchor="end"
            />
            <YAxis hide />
            <Tooltip cursor={{ fill: "rgba(0, 0, 0, 0.1)" }} />
            <Bar dataKey="value" fill="url(#gradient)" barSize={15} />
            <defs>
              <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366F1" />
                <stop offset="100%" stopColor="#3B82F6" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pie Chart */}
      <div className="w-full md:w-1/2 bg-[#202020] p-4 rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-gray-300 font-semibold">Feature Distribution</h3>
          <span className="text-gray-400 text-sm">Feature Breakdown</span>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              fill="#8884d8"
              paddingAngle={5}
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Legend
              iconType="circle"
              layout="horizontal"
              align="center"
              verticalAlign="bottom"
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex justify-around mt-4">
          {pieData.map((entry, index) => (
            <div key={`legend-${index}`} className="text-center">
              <span
                className={`inline-block w-2 h-2 rounded-full`}
                style={{ backgroundColor: entry.color }}
              ></span>
              <span className="text-white font-semibold">{entry.name}</span>
              <p className="text-gray-400 text-sm">{`${Math.round(
                (entry.value /
                  pieData.reduce((sum, item) => sum + item.value, 0)) *
                  100
              )}%`}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LimeCharts;
