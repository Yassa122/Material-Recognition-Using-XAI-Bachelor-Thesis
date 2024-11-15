// pages/data-upload.tsx

import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header"; // Assuming you have a Header component
import UploadData from "@/app/components/UploadData";
import StorageComponent from "@/app/components/StorageComponent"; // New component for storage display
import SmilesDataTable from "@/app/components/SmilesDataTable";

const DataUploadPage = () => {
  return (
    <div className="bg-mainBg min-h-screen text-gray-100 flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 p-6">
        {/* Header */}
        <Header />

        {/* Page Title */}
        <h1 className="text-2xl font-bold mb-6">Data Upload</h1>

        {/* Components Container */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Storage Component */}
          <StorageComponent />

          {/* Upload Component */}
          <div className="bg-[#202020] p-6 rounded-lg shadow-lg flex flex-col justify-center items-center">
            <UploadData />
            <div className="text-gray-300 mt-4 text-center">
              <p>Upload you SMILES data set</p>
              <p className="text-sm text-gray-500 mt-2">
                Stay on the pulse of distributed projects with an online
                whiteboard to plan, coordinate, and discuss
              </p>
              <button className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-lg">
                Start Training now
              </button>
            </div>
          </div>
        </div>
        <SmilesDataTable />
      </div>
    </div>
  );
};

export default DataUploadPage;
