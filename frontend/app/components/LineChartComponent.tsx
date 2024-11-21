// components/LineChartComponent.tsx
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

// Simulated SMILES data with arbitrary properties for visualization
const data = [
  { name: "CCO", propertyA: 85, propertyB: 120 }, // Ethanol
  { name: "CCN(CC)CC", propertyA: 100, propertyB: 160 }, // Diethylamine
  { name: "CC(=O)OC1=CC=CC=C1C(=O)O", propertyA: 130, propertyB: 200 }, // Aspirin
  { name: "C1=CC=C(C=C1)C=O", propertyA: 108, propertyB: 180 }, // Benzaldehyde (highlighted point)
  { name: "C1CCCCC1", propertyA: 150, propertyB: 190 }, // Cyclohexane
  { name: "CCCCCC", propertyA: 170, propertyB: 220 }, // Hexane
];

import { TooltipProps } from "recharts";

const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 text-white p-2 rounded">
        <p className="text-sm">{`SMILES: ${label}`}</p>
        <p className="text-base font-semibold">{`Property A: ${payload[0].value}`}</p>
      </div>
    );
  }
  return null;
};

const LineChartComponent = () => (
  <div className="bg-sidebarBg p-4 rounded-lg">
    <h2 className="text-lg font-semibold text-white mb-2">SMILES Properties</h2>
    <p className="text-green-400 text-sm">Property A &bull; On track</p>
    <p className="text-green-400 mt-1">Highlighted Data Point at Benzaldehyde</p>

    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <XAxis dataKey="name" tick={{ fill: "#888" }} />
        <YAxis hide />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#555" }} />

        {/* Highlighted Dot at Benzaldehyde */}
        <ReferenceDot x="C1=CC=C(C=C1)C=O" y={108} r={5} fill="#4f46e5" stroke="none">
          <text
            x="108"
            y="-10"
            fill="#4f46e5"
            fontSize={12}
            textAnchor="middle"
          >
            108
          </text>
        </ReferenceDot>

        {/* Line with gradient colors */}
        <defs>
          <linearGradient id="colorPropertyA" x1="0" y1="0" x2="1" y2="0">
            <stop offset="5%" stopColor="#4f46e5" stopOpacity={1} />
            <stop offset="95%" stopColor="#06b6d4" stopOpacity={1} />
          </linearGradient>
        </defs>

        <Line
          type="monotone"
          dataKey="propertyA"
          stroke="url(#colorPropertyA)"
          strokeWidth={3}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

export default LineChartComponent;
