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
    // Pagination props
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
    <div className="mb-8">
      <h2 className="text-2xl font-semibold mb-4">Biological Activity Data</h2>
      <div className="overflow-x-auto">
        <table
          {...getTableProps()}
          className="min-w-full bg-gray-700 text-left"
        >
          <thead>
            {headerGroups.map((headerGroup) => (
              <tr {...headerGroup.getHeaderGroupProps()} className="border-b">
                {headerGroup.headers.map((column) => (
                  <th
                    {...column.getHeaderProps(column.getSortByToggleProps())}
                    className="px-4 py-2"
                  >
                    {column.render("Header")}
                    {/* Add a sort indicator */}
                    <span>
                      {column.isSorted
                        ? column.isSortedDesc
                          ? " 🔽"
                          : " 🔼"
                        : ""}
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody {...getTableBodyProps()}>
            {page.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-2 text-center">
                  No data available.
                </td>
              </tr>
            ) : (
              page.map((row) => {
                prepareRow(row);
                return (
                  <tr {...row.getRowProps()} className="border-b">
                    {row.cells.map((cell) => (
                      <td {...cell.getCellProps()} className="px-4 py-2">
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
      <div className="flex justify-between items-center mt-4">
        <div>
          <button
            onClick={() => gotoPage(0)}
            disabled={!canPreviousPage}
            className="px-3 py-1 bg-gray-600 rounded mr-2 disabled:opacity-50"
          >
            {"<<"}
          </button>
          <button
            onClick={() => previousPage()}
            disabled={!canPreviousPage}
            className="px-3 py-1 bg-gray-600 rounded mr-2 disabled:opacity-50"
          >
            {"<"}
          </button>
          <button
            onClick={() => nextPage()}
            disabled={!canNextPage}
            className="px-3 py-1 bg-gray-600 rounded mr-2 disabled:opacity-50"
          >
            {">"}
          </button>
          <button
            onClick={() => gotoPage(pageCount - 1)}
            disabled={!canNextPage}
            className="px-3 py-1 bg-gray-600 rounded disabled:opacity-50"
          >
            {">>"}
          </button>
        </div>
        <span>
          Page{" "}
          <strong>
            {pageIndex + 1} of {pageOptions.length}
          </strong>{" "}
        </span>
        <span>
          | Go to page:{" "}
          <input
            type="number"
            defaultValue={pageIndex + 1}
            onChange={(e) => {
              const page = e.target.value ? Number(e.target.value) - 1 : 0;
              gotoPage(page);
            }}
            className="w-16 px-2 py-1 bg-gray-600 text-gray-100 rounded"
          />
        </span>
        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(Number(e.target.value));
          }}
          className="px-2 py-1 bg-gray-600 text-gray-100 rounded"
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
