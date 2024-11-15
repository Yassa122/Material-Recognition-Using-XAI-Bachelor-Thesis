// components/HorizontalBarChart.tsx
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const data = [
  { name: "Feature 1", value: 350 },
  { name: "Feature 2", value: 320 },
  { name: "Feature 3", value: 300 },
  { name: "Feature 4", value: 280 },
  { name: "Feature 5", value: 250 },
  { name: "Feature 6", value: 220 },
  { name: "Feature 7", value: 200 },
];

const HorizontalBarChart = () => (
  <div className="bg-sidebarBg p-6 rounded-lg shadow-lg">
    <h3 className="text-white text-lg font-semibold mb-4">LIME</h3>
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={[0, 400]} tick={{ fill: "#A0AEC0" }} />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: "#A0AEC0" }}
          width={100}
        />
        <Tooltip cursor={{ fill: "rgba(0, 0, 0, 0.1)" }} />
        <Bar dataKey="value" fill="#4299E1" barSize={20}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill="#4299E1" />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default HorizontalBarChart;
