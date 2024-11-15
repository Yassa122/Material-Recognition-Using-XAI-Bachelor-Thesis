// components/UploadComponent.tsx
const UploadComponent = () => (
  <div className="flex flex-col md:flex-row items-center justify-between bg-sidebarBg p-6 rounded-lg shadow-lg">
    {/* File Upload Box */}
    <div className="border-dashed border-2 border-gray-500 p-8 rounded-lg flex flex-col items-center">
      <p className="text-white font-medium text-lg mb-2">Upload Files</p>
      <p className="text-gray-400 text-sm">CSV and XLSX files are allowed</p>
    </div>

    {/* Description and Button */}
    <div className="mt-4 md:mt-0 md:ml-8 text-center md:text-left">
      <h3 className="text-white font-semibold text-xl">
        Upload your SMILES data set
      </h3>
      <p className="text-gray-400 text-sm mt-2">
        Stay on the pulse of distributed projects with an online whiteboard to
        plan, coordinate, and discuss.
      </p>
      <button className="bg-blue-600 text-white py-2 px-6 rounded mt-4 hover:bg-blue-700">
        Start Training Now
      </button>
    </div>
  </div>
);

export default UploadComponent;
