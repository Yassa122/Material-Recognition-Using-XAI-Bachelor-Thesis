const NewSmilesTable = () => {
  // Example data for the SMILES table
  const smilesData = [
    {
      id: 1,
      molecule: "C1=CC=CC=C1",
      name: "Benzene",
      properties: "Aromatic Hydrocarbon",
    },
    { id: 2, molecule: "CCO", name: "Ethanol", properties: "Alcohol" },
    {
      id: 3,
      molecule: "CC(=O)O",
      name: "Acetic Acid",
      properties: "Carboxylic Acid",
    },
    { id: 4, molecule: "CCN", name: "Ethylamine", properties: "Amine" },
    { id: 5, molecule: "COC", name: "Dimethyl Ether", properties: "Ether" },
  ];

  return (
    <div className="bg-sidebarBg p-6 rounded-xl shadow-lg mt-6">
      <h3 className="text-lg font-bold text-gray-200 mb-4">
        New SMILES Components
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse rounded-lg overflow-hidden">
          {/* Table Header */}
          <thead className="bg-gradient-to-r from-zinc-900 to-zinc-800 text-gray-400">
            <tr>
              <th className="p-4 text-center font-semibold">#</th>
              <th className="p-4 text-center font-semibold">Molecule</th>
              <th className="p-4 text-center font-semibold">Name</th>
            </tr>
          </thead>

          {/* Table Body */}
          <tbody>
            {smilesData.map((item, index) => (
              <tr
                key={item.id}
                className={`${
                  index % 2 === 0 ? "bg-zinc-900" : "bg-zinc-800"
                } hover:bg-zinc-700 transition-all duration-200`}
              >
                <td className="p-4 text-center text-gray-300 font-medium">
                  {item.id}
                </td>
                <td className="p-4 text-center text-blue-400 font-mono">
                  {item.molecule}
                </td>
                <td className="p-4 text-center text-gray-300 font-medium">
                  {item.name}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default NewSmilesTable;
