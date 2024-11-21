import Link from "next/link";

const PredictedPropertiesTable = () => {
  const predictedData = [
    {
      id: 1,
      smiles: "C1=CC=CC=C1",
      name: "Benzene",
      predictedProperties: {
        toxicity: "Low",
        solubility: "High",
        activity: "Active",
      },
    },
    {
      id: 2,
      smiles: "CCO",
      name: "Ethanol",
      predictedProperties: {
        toxicity: "Very Low",
        solubility: "Very High",
        activity: "Moderate",
      },
    },
    {
      id: 3,
      smiles: "CC(=O)O",
      name: "Acetic Acid",
      predictedProperties: {
        toxicity: "Low",
        solubility: "High",
        activity: "Inactive",
      },
    },
    {
      id: 4,
      smiles: "CCN",
      name: "Ethylamine",
      predictedProperties: {
        toxicity: "Moderate",
        solubility: "High",
        activity: "Active",
      },
    },
    {
      id: 5,
      smiles: "COC",
      name: "Dimethyl Ether",
      predictedProperties: {
        toxicity: "Low",
        solubility: "Moderate",
        activity: "Inactive",
      },
    },
  ];

  return (
    <div className="bg-zinc-900 p-6 rounded-xl shadow-lg mt-6">
      <h3 className="text-lg font-bold text-gray-200 mb-4">
        Predicted Properties
      </h3>
      <table className="w-full text-left border-collapse border border-gray-700">
        <thead className="bg-zinc-800 text-gray-400">
          <tr>
            <th className="p-4 border border-gray-700">#</th>
            <th className="p-4 border border-gray-700">SMILES</th>
            <th className="p-4 border border-gray-700">Name</th>
            <th className="p-4 border border-gray-700">Toxicity</th>
            <th className="p-4 border border-gray-700">Solubility</th>
            <th className="p-4 border border-gray-700">Activity</th>
          </tr>
        </thead>
        <tbody>
          {predictedData.map((item, index) => (
            <tr
              key={item.id}
              className={`${
                index % 2 === 0 ? "bg-zinc-800" : "bg-zinc-700"
              } hover:bg-zinc-600`}
            >
              <td className="p-4 border border-gray-700 text-gray-300">
                {item.id}
              </td>
              <td className="p-4 border border-gray-700 text-blue-400 font-mono">
                <Link
                  href={`/pages/compound/${encodeURIComponent(item.smiles)}`}
                  className="hover:underline"
                >
                  {item.smiles}
                </Link>
              </td>
              <td className="p-4 border border-gray-700 text-gray-300">
                {item.name}
              </td>
              <td className="p-4 border border-gray-700 text-green-400 font-medium">
                {item.predictedProperties.toxicity}
              </td>
              <td className="p-4 border border-gray-700 text-yellow-400 font-medium">
                {item.predictedProperties.solubility}
              </td>
              <td className="p-4 border border-gray-700 text-purple-400 font-medium">
                {item.predictedProperties.activity}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default PredictedPropertiesTable;
