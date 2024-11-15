// components/StorageComponent.tsx

import { FaCloud } from "react-icons/fa";
import { FiMoreVertical } from "react-icons/fi";

const StorageComponent = () => {
  return (
    <div className="bg-[#202020] p-6 rounded-lg shadow-lg text-center relative">
      {/* More Options Icon */}
      <div className="absolute top-4 right-4 text-gray-400">
        <FiMoreVertical />
      </div>

      <div className="flex flex-col items-center mb-4">
        {/* Cloud Icon with background circle */}
        <div className="bg-black p-4 rounded-full">
          <FaCloud className="text-blue-500 text-3xl" />
        </div>
        <h2 className="text-xl font-semibold text-white mt-4">Your storage</h2>
        <p className="text-gray-400 text-sm mt-1">
          Supervise your drive space in the easiest way{" "}
        </p>
      </div>

      {/* Storage Bar */}
      <div className="mt-32">
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-[#6C63FF] h-2 rounded-full"
            style={{ width: "50%" }}
          ></div>
        </div>
        <div className="flex justify-between text-gray-400 text-sm mt-2">
          <span>25.6 Gb</span>
          <span>50 Gb</span>
        </div>
      </div>
    </div>
  );
};

export default StorageComponent;
