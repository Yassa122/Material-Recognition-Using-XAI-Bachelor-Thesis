// components/SmilesDataTable.tsx
import { FaEllipsisV } from "react-icons/fa";

const smilesData = [
  {
    name: "Benzene",
    smiles: "C1=CC=CC=C1",
    molecularWeight: "78.11 g/mol",
    meltingPoint: "5.5°C",
    dateAdded: "24.Jan.2021",
  },
  {
    name: "Ethanol",
    smiles: "CCO",
    molecularWeight: "46.07 g/mol",
    meltingPoint: "-114.1°C",
    dateAdded: "12.Jun.2021",
  },
  {
    name: "Methane",
    smiles: "C",
    molecularWeight: "16.04 g/mol",
    meltingPoint: "-182.5°C",
    dateAdded: "5.Jan.2021",
  },
  {
    name: "Acetone",
    smiles: "CC(=O)C",
    molecularWeight: "58.08 g/mol",
    meltingPoint: "-94.7°C",
    dateAdded: "7.Mar.2021",
  },
  {
    name: "Aspirin",
    smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
    molecularWeight: "180.16 g/mol",
    meltingPoint: "135°C",
    dateAdded: "17.Dec.2021",
  },
];

const SmilesDataTable = () => {
  return (
    <div className="bg-[#202020] p-6 rounded-lg shadow-lg mt-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-white">SMILES Dataset</h2>
        <FaEllipsisV className="text-gray-400 cursor-pointer" />
      </div>
      <table className="w-full text-left">
        <thead>
          <tr className="text-gray-500 text-sm border-b border-gray-700">
            <th className="py-2">
              Name <span className="text-xs">▼</span>
            </th>
            <th className="py-2">
              SMILES <span className="text-xs">▼</span>
            </th>
            <th className="py-2">
              Molecular Weight <span className="text-xs">▼</span>
            </th>
            <th className="py-2">
              Melting Point <span className="text-xs">▼</span>
            </th>
            <th className="py-2">
              Date Added <span className="text-xs">▼</span>
            </th>
          </tr>
        </thead>
        <tbody className="text-gray-300 text-sm">
          {smilesData.map((compound, index) => (
            <tr key={index} className="border-b border-gray-700">
              <td className="py-3">{compound.name}</td>
              <td className="py-3">{compound.smiles}</td>
              <td className="py-3">{compound.molecularWeight}</td>
              <td className="py-3">{compound.meltingPoint}</td>
              <td className="py-3">{compound.dateAdded}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SmilesDataTable;
