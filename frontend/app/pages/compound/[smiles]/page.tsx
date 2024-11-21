"use client";

const CompoundDetails = () => {
  // Hardcoded SMILES and compound details
  const smiles = "C1=CC=CC=C1"; // Benzene SMILES
  const data = {
    smiles: smiles,
    twoDImage: "https://via.placeholder.com/300", // Replace with a valid 2D image URL
    threeDImage: "https://via.placeholder.com/300", // Replace with a valid 3D image URL
  };

  return (
    <div className="bg-zinc-900 min-h-screen text-gray-100 p-8">
      <h1 className="text-3xl font-bold mb-6 text-white">
        Compound Details - {data.smiles}
      </h1>
      <div className="grid grid-cols-12 gap-6">
        {/* 2D Molecular Structure */}
        <div className="col-span-6 bg-zinc-800 p-4 rounded-lg shadow-lg">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">
            2D Structure
          </h2>
          <img
            src={data.twoDImage}
            alt="2D Molecular Structure"
            className="rounded"
            style={{ width: "100%", height: "auto" }}
          />
        </div>

        {/* 3D Molecular Structure */}
        <div className="col-span-6 bg-zinc-800 p-4 rounded-lg shadow-lg">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">
            3D Structure
          </h2>
          <img
            src={data.threeDImage}
            alt="3D Molecular Structure"
            className="rounded"
            style={{ width: "100%", height: "auto" }}
          />
        </div>
      </div>
    </div>
  );
};

export default CompoundDetails;
