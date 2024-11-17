import Sidebar from "@/app/components/Sidebar";
import Header from "@/app/components/Header";
import ShapExplanationChart from "@/app/components/ShapExplanationChart";
import ChatComponent from "@/app/components/ChatComponent";
import NewSmilesTable from "@/app/components/NewSmilesTable";

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
        <h1 className="text-2xl font-bold mb-6">Generated Molecules</h1>

        {/* Content Section */}
        <div className="grid grid-cols-12 gap-6">
          {/* SHAP Explanation Chart */}
          <div className="col-span-8 bg-zinc-950 p-6 rounded-xl shadow-lg">
            <ShapExplanationChart />
            <NewSmilesTable />
          </div>

          {/* Chat Component */}
          <div className="col-span-4  rounded-xl shadow-lg">
            <ChatComponent />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUploadPage;
