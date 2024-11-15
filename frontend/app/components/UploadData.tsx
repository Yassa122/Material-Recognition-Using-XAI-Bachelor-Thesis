// components/UploadComponent.tsx

import { FaUpload } from "react-icons/fa";

const UploadData = () => (
  <div className="flex bg-[#202020] p-6 rounded-lg shadow-lg space-x-8">
    {/* Upload Box */}
    <div className="flex-1 border-dashed border-2 bg-black border-gray-500 p-8 rounded-lg flex flex-col items-center justify-center">
      <FaUpload className="text-blue-500 text-5xl mb-4" />
      <p className="text-blue-400 font-semibold text-lg">Upload Files</p>
      <p className="text-gray-400 text-sm mt-1">
        CSV, and XLSX files are allowed
      </p>
    </div>
  </div>
);

export default UploadData;
