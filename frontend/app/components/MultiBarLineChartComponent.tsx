// components/MultiBarLineChartComponent.tsx
import React from "react";
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
} from "recharts";

// Example SMILES data with arbitrary properties
const data = [
  { name: "CCO", A: 85, B: 120, C: 90 }, // Ethanol
  { name: "CCN(CC)CC", A: 100, B: 160, C: 110 }, // Diethylamine
  { name: "CC(=O)OC1=CC=CC=C1C(=O)O", A: 130, B: 200, C: 140 }, // Aspirin
  { name: "C1=CC=C(C=C1)C=O", A: 75, B: 95, C: 80 }, // Benzaldehyde
  { name: "CC(=O)NC1=CC=CC=C1", A: 95, B: 130, C: 100 }, // Acetanilide
  { name: "C1CCCCC1", A: 110, B: 150, C: 120 }, // Cyclohexane
  { name: "CCCCCC", A: 60, B: 80, C: 70 }, // Hexane
  { name: "CC1=CC=CC=C1", A: 125, B: 170, C: 130 }, // Toluene
];

import { TooltipProps } from "recharts";

const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 text-white p-2 rounded shadow-md">
        <p className="font-semibold">{label} (SMILES)</p>
        {payload.map((item: { name: string; value: number; color: string }, index: number) => (
          <p key={index} style={{ color: item.color }}>
            {item.name}: {item.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const MultiBarLineChartComponent = () => (
  <div className="bg-sidebarBg p-4 rounded-lg">
    <div className="flex justify-between items-center mb-2">
      <p className="text-gray-300 font-semibold">Compound Properties</p>
      <div className="bg-gray-700 p-2 rounded-full">
        {/* Icon or additional options here */}
      </div>
    </div>

    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
        <XAxis dataKey="name" tick={{ fill: "#888" }} />
        <YAxis yAxisId="left" orientation="left" tick={{ fill: "#888" }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: "#888" }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend verticalAlign="top" align="right" iconType="circle" />

        {/* Bar Components for Properties A, B, C */}
        <Bar yAxisId="left" dataKey="A" fill="#4F86E5" barSize={20} />
        <Bar yAxisId="left" dataKey="B" fill="#4FC3E5" barSize={20} />
        <Bar yAxisId="left" dataKey="C" fill="#88B4E5" barSize={20} />

        {/* Line Component for trend of Property C */}
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="C"
          stroke="#E57373"
          strokeWidth={2}
        />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default MultiBarLineChartComponent;
