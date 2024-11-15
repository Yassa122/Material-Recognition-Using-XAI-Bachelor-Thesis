// components/StackedBarChart.tsx
import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Simulated SMILES data with arbitrary properties for visualization
const data = [
  { name: "CCO", solubility: 300, meltingPoint: 200, molecularWeight: 100 }, // Ethanol
  {
    name: "CCN(CC)CC",
    solubility: 250,
    meltingPoint: 180,
    molecularWeight: 150,
  }, // Diethylamine
  {
    name: "CC(=O)OC1=CC=CC=C1C(=O)O",
    solubility: 220,
    meltingPoint: 250,
    molecularWeight: 200,
  }, // Aspirin
  {
    name: "C1=CC=C(C=C1)C=O",
    solubility: 190,
    meltingPoint: 210,
    molecularWeight: 170,
  }, // Benzaldehyde
  {
    name: "C1CCCCC1",
    solubility: 160,
    meltingPoint: 220,
    molecularWeight: 130,
  }, // Cyclohexane
  { name: "CCCCCC", solubility: 140, meltingPoint: 180, molecularWeight: 120 }, // Hexane
];

const StackedBarChart = () => (
  <div className="bg-sidebarBg p-6 rounded-lg shadow-lg">
    <h3 className="text-white text-lg font-semibold mb-4">SMILES Properties</h3>
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={data}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ fill: "#A0AEC0" }} />
        <YAxis tick={{ fill: "#A0AEC0" }} />
        <Tooltip />
        <Legend />

        {/* Each property of SMILES data is stacked */}
        <Bar
          dataKey="solubility"
          stackId="a"
          fill="#4FD1C5"
          name="Solubility"
        />
        <Bar
          dataKey="meltingPoint"
          stackId="a"
          fill="#63B3ED"
          name="Melting Point"
        />
        <Bar
          dataKey="molecularWeight"
          stackId="a"
          fill="#805AD5"
          name="Molecular Weight"
        />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default StackedBarChart;
