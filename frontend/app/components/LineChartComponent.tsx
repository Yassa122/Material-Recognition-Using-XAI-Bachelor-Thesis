// components/LineChartComponent.tsx

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  Legend,
} from "recharts";
import { TooltipProps } from "recharts";
import axios from "axios"; // Using axios for HTTP requests

// Define the shape of your chart data
interface ChartData {
  name: string;
  propertyA: number;
  propertyB: number;
  propertyC: number;
}

// Custom Tooltip Component
const CustomTooltip = ({
  active,
  payload,
  label,
}: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 text-white p-2 rounded">
        <p className="text-sm">{`SMILES: ${label}`}</p>
        <p className="text-base font-semibold">{`Property A: ${payload[0].value}`}</p>
        <p className="text-base font-semibold">{`Property B: ${payload[1].value}`}</p>
        <p className="text-base font-semibold">{`Property C: ${payload[2].value}`}</p>
      </div>
    );
  }
  return null;
};

const LineChartComponent = () => {
  const [data, setData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await axios.get<ChartData[]>(
          "http://127.0.0.1:5000/api/chart-data"
        );
        setData(response.data);
        setLoading(false);
      } catch (err: any) {
        console.error("Error fetching chart data:", err);
        setError(err.response?.data?.error || "Failed to fetch chart data.");
        setLoading(false);
      }
    };

    fetchChartData();
  }, []);

  // Function to find the data point with the highest propertyA
  const getHighlightedPoint = (): ChartData | undefined => {
    if (data.length === 0) return undefined;
    return data.reduce((prev, current) =>
      prev.propertyA > current.propertyA ? prev : current
    );
  };

  if (loading) {
    return (
      <div className="bg-sidebarBg p-4 rounded-lg flex justify-center items-center">
        <div className="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-sidebarBg p-4 rounded-lg text-red-500">
        <p>Error: {error}</p>
      </div>
    );
  }

  const highlightedPoint = getHighlightedPoint();

  return (
    <div className="bg-sidebarBg p-4 rounded-lg">
      <h2 className="text-lg font-semibold text-white mb-2">
        SMILES Properties
      </h2>
      <p className="text-green-400 text-sm">Properties A, B & C</p>
      <p className="text-green-400 mt-1">
        Highlighted Data Point at Benzaldehyde
      </p>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <XAxis dataKey="name" tick={{ fill: "#888" }} />
          <YAxis hide />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#555" }} />
          <Legend verticalAlign="top" height={36} />

          {/* Highlighted Dot if Exists */}
          {highlightedPoint && (
            <ReferenceDot
              x={highlightedPoint.name}
              y={highlightedPoint.propertyA}
              r={5}
              fill="#4f46e5"
              stroke="none"
              label={{
                position: "top",
                value: highlightedPoint.propertyA,
                fill: "#4f46e5",
                fontSize: 12,
                fontWeight: "bold",
              }}
            />
          )}

          {/* Line for Property A */}
          <Line
            type="monotone"
            dataKey="propertyA"
            name="Predicted pIC50"
            stroke="#4f46e5"
            strokeWidth={3}
            dot={false}
          />

          {/* Line for Property B */}
          <Line
            type="monotone"
            dataKey="propertyB"
            name="Predicted logP"
            stroke="#06b6d4"
            strokeWidth={3}
            dot={false}
          />

          {/* Line for Property C */}
          <Line
            type="monotone"
            dataKey="propertyC"
            name="Predicted Num Atoms"
            stroke="#f97316"
            strokeWidth={3}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChartComponent;
