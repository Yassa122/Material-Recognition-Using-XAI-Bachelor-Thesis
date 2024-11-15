// components/LimeCharts.tsx
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

// Example bar data for SMILES string occurrences at different time points
const barData = [
  { time: "00", count: 20 }, // Example usage count at 00:00
  { time: "04", count: 35 },
  { time: "08", count: 50 },
  { time: "12", count: 70 },
  { time: "14", count: 55 },
  { time: "16", count: 80 },
  { time: "18", count: 40 },
];

// Example pie data for compound types in a dataset
const pieData = [
  { name: "Aromatic", value: 45, color: "#4F46E5" }, // Aromatic compounds
  { name: "Aliphatic", value: 30, color: "#3B82F6" }, // Aliphatic compounds
  { name: "Heterocyclic", value: 25, color: "#E5E7EB" }, // Heterocyclic compounds
];

const LimeCharts = () => (
  <div className="flex flex-col md:flex-row space-y-6 md:space-y-0 md:space-x-6 bg-sidebarBg p-6 rounded-lg shadow-lg">
    {/* Bar Chart */}
    <div className="w-full md:w-1/2 bg-[#202020] p-4 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-gray-300 font-semibold">SMILES Usage Over Time</h3>
        <span className="text-green-400 font-semibold text-sm">+15%</span>
      </div>
      <p className="text-3xl font-bold text-white mb-1">SMILES Data</p>
      <p className="text-gray-400 text-sm mb-6">Usage Count</p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={barData}>
          <XAxis dataKey="time" tick={{ fill: "#A0AEC0" }} />
          <YAxis hide />
          <Tooltip cursor={{ fill: "rgba(0, 0, 0, 0.1)" }} />
          <Bar dataKey="count" fill="url(#gradient)" barSize={15} />
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
        <h3 className="text-gray-300 font-semibold">
          Compound Type Distribution
        </h3>
        <span className="text-gray-400 text-sm">Dataset Breakdown</span>
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
        <div className="text-center">
          <span className="inline-block w-2 h-2 rounded-full bg-[#4F46E5] mr-2"></span>
          <span className="text-white font-semibold">Aromatic</span>
          <p className="text-gray-400 text-sm">45%</p>
        </div>
        <div className="text-center">
          <span className="inline-block w-2 h-2 rounded-full bg-[#3B82F6] mr-2"></span>
          <span className="text-white font-semibold">Aliphatic</span>
          <p className="text-gray-400 text-sm">30%</p>
        </div>
        <div className="text-center">
          <span className="inline-block w-2 h-2 rounded-full bg-[#E5E7EB] mr-2"></span>
          <span className="text-white font-semibold">Heterocyclic</span>
          <p className="text-gray-400 text-sm">25%</p>
        </div>
      </div>
    </div>
  </div>
);

export default LimeCharts;
