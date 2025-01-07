"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import Viewer3D from "./Viewer3D";
import NetworkGraph from "./NetworkGraph";
import SmilesDrawer from "smiles-drawer";
import { data } from "framer-motion/client";

interface MoleculeData {
  atoms: { id: number; element: string }[];
  bonds: { startAtomIndex: number; endAtomIndex: number }[];
}

const CompoundDetails = () => {
  const [moleculeData, setMoleculeData] = useState<MoleculeData | null>(null);
  const { smiles: rawSmiles } = useParams();
  const smiles = typeof rawSmiles === "string" ? decodeURIComponent(rawSmiles) : "";

  const [sdfData, setSdfData] = useState(null);
  // const [moleculeData, setMoleculeData] = useState(null);
  const [smilesTree, setSmilesTree] = useState(null); // New state for the SMILES tree
  const canvasRef = useRef(null);
  const viewerRef = useRef(null);

  // Fetch SDF data (3D structure)
  useEffect(() => {
    if (!smiles) {
      console.error("SMILES string is undefined or empty.");
      return;
    }

    const fetchSdfData = async () => {
      try {
        const response = await fetch(
          `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(
            smiles
          )}/SDF?record_type=3d`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch SDF data");
        }
        const sdfText = await response.text();
        setSdfData(sdfText);
      } catch (error) {
        console.error("Error fetching SDF data:", error);
      }
    };

    fetchSdfData();
  }, [smiles]);

  // Parse SMILES and set molecule data
  useEffect(() => {
    if (!smiles) {
      console.error("SMILES string is undefined or empty.");
      return;
    }

    console.log("Attempting to parse SMILES:", smiles);

    SmilesDrawer.parse(
      smiles,
      (tree) => {
        const atoms = Object.values(tree.nodes).map((node: any) => ({
          id: node.id,
          element: node.atom,
        }));

        const bonds = tree.edges.map((edge: { sourceId: any; targetId: any; }) => ({
          startAtomIndex: edge.sourceId,
          endAtomIndex: edge.targetId,
        }));

        setMoleculeData({ atoms, bonds });
        setSmilesTree(tree); // Store the SMILES tree for drawing
        console.log("SMILES parsed successfully", { atoms, bonds });
      },
      (err) => {
        console.error("Error parsing SMILES:", err);
        setMoleculeData(null);
        setSmilesTree(null);
      }
    );
  }, [smiles]);

  // Draw on canvas once smilesTree and canvasRef are available
  useEffect(() => {
    if (smilesTree && canvasRef.current) {
      console.log("Drawing on canvas...");
      const drawer = new SmilesDrawer.Drawer({
        width: 400,
        height: 400,
      });
      drawer.draw(smilesTree, canvasRef.current, "light", false);
    } else {
      if (!smilesTree) {
        console.error("smilesTree is null");
      }
      if (!canvasRef.current) {
        console.error("canvasRef.current is null");
      }
    }
  }, [smilesTree, canvasRef]);

  return (
    <div className="bg-zinc-900 min-h-screen text-gray-100 p-8">
      <h1 className="text-3xl font-bold mb-6 text-white">
        Compound Details - {smiles}
      </h1>
      <div className="grid grid-cols-12 gap-6">
        {/* 2D Molecular Structure */}
        <div className="col-span-6 bg-zinc-800 p-4 rounded-lg shadow-lg">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">
            2D Structure
          </h2>
          <canvas
            ref={canvasRef}
            width={400}
            height={400}
            style={{ width: "100%", height: "auto" }}
          ></canvas>
          {moleculeData ? (
            <NetworkGraph molecule={moleculeData} />
          ) : (
            <p>Error loading 2D structure.</p>
          )}
        </div>

        {/* 3D Molecular Structure */}
        <div className="col-span-6 bg-zinc-800 p-4 rounded-lg shadow-lg flex flex-col items-center">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">
            3D Structure
          </h2>
          {sdfData ? (
            <>
              <div
                ref={viewerRef}
                style={{
                  width: "100%",
                  height: "400px",
                  position: "relative",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              />
              <Viewer3D sdf={sdfData} viewerRef={viewerRef} />
            </>
          ) : (
            <p>Error loading 3D structure.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default CompoundDetails;
