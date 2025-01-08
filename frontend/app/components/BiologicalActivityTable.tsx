// components/BiologicalActivityTable.js
import { useMemo } from "react";
import { useTable, usePagination, useSortBy } from "react-table";

const BiologicalActivityTable = ({ data }) => {
  const columns = useMemo(
    () => [
      {
        Header: "SMILES",
        accessor: "SMILES",
      },
      {
        Header: "MolWt",
        accessor: "MolWt",
      },
      {
        Header: "Num H Donors",
        accessor: "NumHDonors",
      },
      {
        Header: "Num H Acceptors",
        accessor: "NumHAcceptors",
      },
      {
        Header: "Predicted logP",
        accessor: "Predicted_logP",
      },
      {
        Header: "Estimated Binding Affinity",
        accessor: "Estimated_Binding_Affinity",
      },
      {
        Header: "Error",
        accessor: "error",
      },
    ],
    []
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    prepareRow,
    page,
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: { pageIndex, pageSize },
  } = useTable(
    {
      columns,
      data,
      initialState: { pageIndex: 0 }, // Start from first page
    },
    useSortBy,
    usePagination
  );

  return (
    <div className="mb-8 bg-zinc-900 text-gray-200 p-6 rounded-md shadow-md">
      <h2 className="text-2xl font-bold mb-6">Biological Activity Data</h2>

      <div className="overflow-x-auto rounded-md border border-zinc-800">
        <table {...getTableProps()} className="w-full border-collapse">
          <thead className="bg-zinc-800">
            {headerGroups.map((headerGroup) => (
              <tr
                {...headerGroup.getHeaderGroupProps()}
                className="border-b border-zinc-700"
              >
                {headerGroup.headers.map((column) => (
                  <th
                    {...column.getHeaderProps(column.getSortByToggleProps())}
                    className="px-4 py-3 text-left text-sm font-semibold uppercase tracking-wide border-zinc-700"
                  >
                    {column.render("Header")}
                    <span className="ml-1">
                      {column.isSorted ? (column.isSortedDesc ? "▼" : "▲") : ""}
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>

          <tbody {...getTableBodyProps()} className="text-sm">
            {page.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-3 text-center border-b border-zinc-700"
                >
                  No data available.
                </td>
              </tr>
            ) : (
              page.map((row) => {
                prepareRow(row);
                return (
                  <tr
                    {...row.getRowProps()}
                    className="border-b border-zinc-700 hover:bg-zinc-800"
                  >
                    {row.cells.map((cell) => (
                      <td {...cell.getCellProps()} className="px-4 py-3">
                        {cell.render("Cell") || "-"}
                      </td>
                    ))}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-center mt-4 gap-2">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => gotoPage(0)}
            disabled={!canPreviousPage}
            className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-50"
          >
            {"<<"}
          </button>
          <button
            onClick={() => previousPage()}
            disabled={!canPreviousPage}
            className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-50"
          >
            {"<"}
          </button>
          <button
            onClick={() => nextPage()}
            disabled={!canNextPage}
            className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-50"
          >
            {">"}
          </button>
          <button
            onClick={() => gotoPage(pageCount - 1)}
            disabled={!canNextPage}
            className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-50"
          >
            {">>"}
          </button>
        </div>

        <div className="text-sm">
          Page{" "}
          <strong>
            {pageIndex + 1} of {pageOptions.length}
          </strong>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-sm">Go to page:</span>
          <input
            type="number"
            defaultValue={pageIndex + 1}
            onChange={(e) => {
              const page = e.target.value ? Number(e.target.value) - 1 : 0;
              gotoPage(page);
            }}
            className="w-16 px-2 py-1 bg-zinc-800 text-gray-200 rounded"
          />
        </div>

        <select
          value={pageSize}
          onChange={(e) => setPageSize(Number(e.target.value))}
          className="px-2 py-1 bg-zinc-800 text-gray-200 rounded"
        >
          {[5, 10, 20, 50].map((pageSizeOption) => (
            <option key={pageSizeOption} value={pageSizeOption}>
              Show {pageSizeOption}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default BiologicalActivityTable;
