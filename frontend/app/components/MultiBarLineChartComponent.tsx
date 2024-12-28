// components/DynamicMultiBarLineChart.tsx

import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceDot,
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
      <div className="bg-gray-800 text-white p-2 rounded shadow-md">
        <p className="font-semibold">{`${label} (SMILES)`}</p>
        {payload.map(
          (
            item: { name: string; value: number; color: string },
            index: number
          ) => (
            <p key={index} style={{ color: item.color }}>
              {`${item.name}: ${item.value}`}
            </p>
          )
        )}
      </div>
    );
  }
  return null;
};

const DynamicMultiBarLineChart = () => {
  const [data, setData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await axios.get<ChartData[]>(
          "http://127.0.0.1:5000/api/compound-properties"
        );
        console.log("Fetched Chart Data:", response.data); // Debugging line
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

  // Function to find the data point with the highest propertyC
  const getHighlightedPoint = (): ChartData | undefined => {
    if (data.length === 0) return undefined;
    return data.reduce((prev, current) =>
      prev.propertyC > current.propertyC ? prev : current
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
      <div className="flex justify-between items-center mb-2">
        <p className="text-gray-300 font-semibold">Compound Properties</p>
        <div className="bg-gray-700 p-2 rounded-full">
          {/* You can add an icon or additional options here */}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
          {/* Explicitly assign xAxisId="0" */}
          <XAxis dataKey="name" xAxisId="0" tick={{ fill: "#888" }} />

          <YAxis yAxisId="left" orientation="left" tick={{ fill: "#888" }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: "#888" }} />

          <Tooltip content={<CustomTooltip />} />
          <Legend verticalAlign="top" align="right" iconType="circle" />

          {/* Bars use yAxisId="left" */}
          <Bar
            yAxisId="left"
            dataKey="propertyA"
            name="Predicted pIC50"
            fill="#4F86E5"
            barSize={20}
          />
          <Bar
            yAxisId="left"
            dataKey="propertyB"
            name="Predicted logP"
            fill="#4FC3E5"
            barSize={20}
          />
          <Bar
            yAxisId="left"
            dataKey="propertyC"
            name="Predicted Num Atoms"
            fill="#88B4E5"
            barSize={20}
          />

          {/* The line uses yAxisId="right" for propertyC trend */}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="propertyC"
            name="Property C Trend"
            stroke="#E57373"
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />

          {/* ReferenceDot must match xAxisId and yAxisId of the line axis */}
          {highlightedPoint && (
            <ReferenceDot
              xAxisId="0"
              yAxisId="right"
              x={highlightedPoint.name}
              y={highlightedPoint.propertyC}
              r={5}
              fill="#FFC107"
              stroke="none"
              label={{
                position: "top",
                value: highlightedPoint.propertyC,
                fill: "#FFC107",
                fontSize: 12,
                fontWeight: "bold",
              }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DynamicMultiBarLineChart;
