// components/BiologicalActivityCharts.js
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  Legend,
  LineChart,
  Line,
} from "recharts";

const BiologicalActivityCharts = ({ data }) => {
  // Prepare data for Binding Affinity Distribution (Histogram-like)
  const bindingAffinityData = data
    .map((item) => ({
      bindingAffinity: item.Estimated_Binding_Affinity,
    }))
    .filter((item) => item.bindingAffinity !== null);

  // Prepare data for MolWt vs Binding Affinity
  const molWtBindingAffinityData = data
    .map((item) => ({
      MolWt: item.MolWt,
      BindingAffinity: item.Estimated_Binding_Affinity,
    }))
    .filter((item) => item.MolWt !== null && item.BindingAffinity !== null);

  // Prepare data for logP vs Binding Affinity
  const logPBindingAffinityData = data
    .map((item) => ({
      logP: item.Predicted_logP,
      BindingAffinity: item.Estimated_Binding_Affinity,
    }))
    .filter((item) => item.logP !== null && item.BindingAffinity !== null);

  // Group binding affinities into bins for bar chart
  const affinityBins = {};
  bindingAffinityData.forEach((item) => {
    const bin = Math.floor(item.bindingAffinity);
    affinityBins[bin] = (affinityBins[bin] || 0) + 1;
  });

  const affinityBarData = Object.keys(affinityBins)
    .map((bin) => ({
      bindingAffinity: bin,
      count: affinityBins[bin],
    }))
    .sort((a, b) => a.bindingAffinity - b.bindingAffinity);

  return (
    <div className="space-y-8">
      {/* Binding Affinity Distribution */}
      <div>
        <h3 className="text-xl font-semibold mb-4">
          Binding Affinity Distribution
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={affinityBarData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="bindingAffinity"
              label={{
                value: "Binding Affinity",
                position: "insideBottom",
                offset: -5,
              }}
            />
            <YAxis
              label={{ value: "Count", angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#4CAF50" name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* MolWt vs Binding Affinity Scatter Plot */}
      <div>
        <h3 className="text-xl font-semibold mb-4">
          MolWt vs Binding Affinity
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid />
            <XAxis
              type="number"
              dataKey="MolWt"
              name="Molecular Weight"
              label={{ value: "MolWt", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              type="number"
              dataKey="BindingAffinity"
              name="Binding Affinity"
              label={{
                value: "Binding Affinity",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Legend />
            <Scatter
              name="Compounds"
              data={molWtBindingAffinityData}
              fill="#8884d8"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* logP vs Binding Affinity Scatter Plot */}
      <div>
        <h3 className="text-xl font-semibold mb-4">logP vs Binding Affinity</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid />
            <XAxis
              type="number"
              dataKey="logP"
              name="logP"
              label={{ value: "logP", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              type="number"
              dataKey="BindingAffinity"
              name="Binding Affinity"
              label={{
                value: "Binding Affinity",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Legend />
            <Scatter
              name="Compounds"
              data={logPBindingAffinityData}
              fill="#82ca9d"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default BiologicalActivityCharts;
