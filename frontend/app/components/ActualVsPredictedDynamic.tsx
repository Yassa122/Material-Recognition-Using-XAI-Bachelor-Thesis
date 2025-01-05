import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  TooltipProps,
} from "recharts";

interface DataPoint {
  SMILES: string;
  actual: number;
  predicted: number;
}

interface PropertyEntry {
  data: DataPoint[];
  mse: number | null;
  mae: number | null;
  r2: number | null;
}

interface DynamicResponse {
  properties: {
    [propertyName: string]: PropertyEntry;
  };
}

/**
 * Helper to color the scatter points based on how large the error is.
 * For a more dynamic or smooth scale, consider using a color interpolation library (like d3-scale).
 */
const getColorByError = (error: number) => {
  const absErr = Math.abs(error);

  // Example thresholds, tweak them as you wish:
  if (absErr > 1) {
    return "#ff0000"; // Red for large error
  } else if (absErr > 0.5) {
    return "#ff8c00"; // Dark orange
  } else if (absErr > 0.2) {
    return "#ffd700"; // Gold
  }
  return "#82ca9d"; // Default green
};

/**
 * Custom Tooltip to show more info about each data point
 */
const CustomTooltip: React.FC<TooltipProps<number, string>> = (props) => {
  const { active, payload } = props;

  if (!active || !payload || !payload.length) {
    return null;
  }

  const { x: actual, y: predicted, SMILES } = payload[0].payload;
  const error = predicted - actual;

  return (
    <div
      style={{ background: "#fff", padding: "8px", border: "1px solid #ccc" ,font: "black"}}
    >
      <p>
        <strong>SMILES:</strong> {SMILES}
      </p>
      <p>
        <strong>Actual:</strong> {actual.toFixed(4)}
      </p>
      <p>
        <strong>Predicted:</strong> {predicted.toFixed(4)}
      </p>
      <p style={{ color: getColorByError(error) }}>
        <strong>Error:</strong> {(predicted - actual).toFixed(4)}
      </p>
    </div>
  );
};

const ActualVsPredictedScatter: React.FC = () => {
  const [responseData, setResponseData] = useState<DynamicResponse | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch data on mount
  useEffect(() => {
    axios
      .get<DynamicResponse>(
        "http://localhost:5000/api/actual_vs_predicted_dynamic"
      )
      .then((res) => setResponseData(res.data))
      .catch((err) => {
        console.error("Error fetching actual vs. predicted data:", err);
        setError("Failed to fetch actual vs. predicted data.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-white">Loading scatter plots...</div>;
  }

  if (error) {
    return <div className="text-red-500">{error}</div>;
  }

  if (!responseData) {
    return <div className="text-white">No data available.</div>;
  }

  const { properties } = responseData;
  const propNames = Object.keys(properties);

  if (propNames.length === 0) {
    return <div className="text-white">No properties found in response.</div>;
  }

  // Utility: convert raw data to Recharts data points
  const convertData = (entries: DataPoint[]) =>
    entries.map((row) => ({
      x: row.actual,
      y: row.predicted,
      SMILES: row.SMILES,
      error: row.predicted - row.actual, // handy for color mapping
    }));

  // Creates a diagonal line from [minVal, minVal] to [maxVal, maxVal]
  const createDiagonalLine = (minVal: number, maxVal: number) =>
    [minVal, maxVal].map((val) => ({ x: val, y: val }));

  return (
    <div className="space-y-8">
      <h2 className="text-white text-xl font-bold mb-4">
        Actual vs. Predicted
      </h2>
      {propNames.map((propName) => {
        const { data, mse, mae, r2 } = properties[propName];
        const chartData = convertData(data);

        const allActual = chartData.map((d) => d.x);
        const allPred = chartData.map((d) => d.y);
        const minVal = Math.min(...allActual, ...allPred);
        const maxVal = Math.max(...allActual, ...allPred);

        // Expand domain slightly so points aren't squashed against the axis
        const marginFactor = 0.1;
        const range = maxVal - minVal;
        const domainMin = minVal - range * marginFactor;
        const domainMax = maxVal + range * marginFactor;

        const diagonalData = createDiagonalLine(domainMin, domainMax);

        return (
          <div key={propName} className="bg-neutral-800 p-4 rounded-md">
            <h3 className="text-white font-semibold mb-2 capitalize">
              {propName} Scatter
            </h3>
            <p className="text-gray-300 mb-3">
              <span className="pr-3">
                <strong>MSE:</strong> {mse?.toFixed(4) || "N/A"}
              </span>
              <span className="pr-3">
                <strong>MAE:</strong> {mae?.toFixed(4) || "N/A"}
              </span>
              <span className="pr-3">
                <strong>R²:</strong> {r2?.toFixed(4) || "N/A"}
              </span>
            </p>

            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#666" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Actual"
                    domain={[domainMin, domainMax]}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Predicted"
                    domain={[domainMin, domainMax]}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ strokeDasharray: "3 3" }}
                  />
                  <Legend />

                  {/* Diagonal reference (perfect correlation) */}
                  <Scatter
                    name="Perfect Correlation"
                    data={diagonalData}
                    fill="rgba(255,255,255,0.5)"
                    line
                    shape="none"
                  />

                  {/* Actual vs Predicted scatter points with color-coding */}
                  <Scatter
                    name={propName}
                    data={chartData}
                    // Instead of a static color, use a function
                    shape={(props) => {
                      const { cx, cy, payload } = props;
                      const pointColor = getColorByError(payload.error);
                      return <circle cx={cx} cy={cy} r={4} fill={pointColor} />;
                    }}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ActualVsPredictedScatter;
