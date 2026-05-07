import { useState } from "react";
import { Table, ChevronDown, ChevronRight } from "lucide-react";

export default function DataTable({ data }) {
  const [open, setOpen] = useState(false);
  const { columns, rows, row_count, truncated } = data;

  return (
    <div className="mt-2 border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800/40 transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Table size={12} />
        <span>
          Result Data ({row_count.toLocaleString()} rows
          {truncated ? ", showing first 500" : ""})
        </span>
      </button>
      {open && (
        <div className="overflow-x-auto max-h-80 overflow-y-auto border-t border-gray-800">
          <table className="w-full text-xs">
            <thead className="sticky top-0">
              <tr>
                {columns.map((col, i) => (
                  <th
                    key={i}
                    className="px-3 py-2 bg-gray-800 text-left text-gray-300 font-semibold whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className="border-t border-gray-800/50 hover:bg-gray-800/30"
                >
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className="px-3 py-1.5 text-gray-400 whitespace-nowrap"
                    >
                      {cell === null ? (
                        <span className="text-gray-600 italic">null</span>
                      ) : typeof cell === "number" ? (
                        cell.toLocaleString()
                      ) : (
                        String(cell)
                      )}
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
