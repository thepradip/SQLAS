import { useState } from "react";
import { Table, ChevronDown, ChevronRight, Download } from "lucide-react";

function formatCell(cell, colName) {
  if (cell === null || cell === undefined)
    return <span className="text-gray-600 italic">null</span>;
  if (typeof cell === "number") {
    // Large numbers get commas; decimals stay short
    return cell % 1 === 0
      ? cell.toLocaleString()
      : cell.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return String(cell);
}

function isNumericColumn(rows, colIndex) {
  return rows.some((r) => typeof r[colIndex] === "number");
}

export default function DataTable({ data }) {
  const [open, setOpen] = useState(true);
  const { columns, rows, row_count, truncated } = data;

  const handleExportCSV = () => {
    const header = columns.join(",");
    const body = rows
      .map((r) =>
        r.map((c) => (c === null ? "" : `"${String(c).replace(/"/g, '""')}"`)).join(",")
      )
      .join("\n");
    const blob = new Blob([header + "\n" + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "query_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-3 border border-gray-700/60 rounded-xl overflow-hidden bg-gray-900/50 shadow-lg shadow-black/20">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-gray-800/60 border-b border-gray-700/40">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 text-xs text-gray-300 hover:text-white transition-colors"
        >
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <Table size={13} className="text-indigo-400" />
          <span className="font-medium">
            Result Data
          </span>
          <span className="text-gray-500 font-normal">
            — {row_count.toLocaleString()} row{row_count !== 1 ? "s" : ""}
            {truncated ? " (showing first 500)" : ""}
          </span>
        </button>
        {open && (
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 transition-colors"
            title="Export as CSV"
          >
            <Download size={11} />
            CSV
          </button>
        )}
      </div>

      {/* Table */}
      {open && (
        <div className="overflow-x-auto max-h-[420px] overflow-y-auto datatable-scroll">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 z-10">
              <tr>
                <th className="px-3 py-2.5 bg-gray-800 text-center text-gray-500 font-medium w-10 border-b border-gray-700/50">
                  #
                </th>
                {columns.map((col, i) => (
                  <th
                    key={i}
                    className={`px-4 py-2.5 bg-gray-800 font-semibold whitespace-nowrap border-b border-gray-700/50 ${
                      isNumericColumn(rows, i) ? "text-right text-indigo-300" : "text-left text-gray-200"
                    }`}
                  >
                    {col.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className={`border-b border-gray-800/40 transition-colors hover:bg-indigo-950/20 ${
                    i % 2 === 0 ? "bg-gray-900/30" : "bg-transparent"
                  }`}
                >
                  <td className="px-3 py-2 text-center text-gray-600 font-mono tabular-nums">
                    {i + 1}
                  </td>
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className={`px-4 py-2 whitespace-nowrap tabular-nums ${
                        isNumericColumn(rows, j)
                          ? "text-right text-gray-300 font-mono"
                          : "text-left text-gray-400"
                      }`}
                    >
                      {formatCell(cell, columns[j])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
