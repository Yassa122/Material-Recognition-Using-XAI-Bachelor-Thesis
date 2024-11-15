// components/TrainingStatusTable.tsx
import {
  FaCheckCircle,
  FaTimesCircle,
  FaExclamationTriangle,
} from "react-icons/fa";

const data = [
  {
    name: "Dataset.csv",
    status: "Completed",
    date: "18 Apr 2024",
    progress: 100,
    icon: <FaCheckCircle className="text-green-500" />,
  },
  {
    name: "Dataset2.csv",
    status: "Error",
    date: "18 Apr 2024",
    progress: 50,
    icon: <FaTimesCircle className="text-red-500" />,
  },
  {
    name: "Dataset3.csv",
    status: "In progress",
    date: "20 May 2024",
    progress: 75,
    icon: <FaExclamationTriangle className="text-yellow-500" />,
  },
  {
    name: "Dataset4.csv",
    status: "Completed",
    date: "12 Jul 2024",
    progress: 100,
    icon: <FaCheckCircle className="text-green-500" />,
  },
];

const TrainingStatusTable = () => {
  return (
    <div className="bg-[#202020] p-6 rounded-lg shadow-lg">
      <h3 className="text-white text-lg font-semibold mb-4">Training Status</h3>
      <table className="w-full text-left text-gray-300">
        <thead>
          <tr className="border-b border-gray-600">
            <th className="py-2">Name</th>
            <th className="py-2">Status</th>
            <th className="py-2">Date</th>
            <th className="py-2">Progress</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index} className="border-b border-gray-800">
              <td className="py-3">{row.name}</td>
              <td className="py-3 flex items-center space-x-2">
                {row.icon}
                <span>{row.status}</span>
              </td>
              <td className="py-3">{row.date}</td>
              <td className="py-3">
                <div className="relative w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${row.progress}%`,
                      backgroundColor:
                        row.progress === 100 ? "#6366F1" : "#3B82F6",
                    }}
                  ></div>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TrainingStatusTable;
