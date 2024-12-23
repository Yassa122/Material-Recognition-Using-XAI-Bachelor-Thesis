import React, { useEffect } from "react";

export default function Viewer3D({ sdf, viewerRef }) {
    useEffect(() => {
        if (!sdf) {
            console.error("SDF data is missing!");
            return;
        }

        const initializeViewer = () => {
            try {
                if (window.$3Dmol && viewerRef.current) {
                    const viewer = $3Dmol.createViewer(viewerRef.current, {
                        backgroundColor: "#2C2F33",
                    });
                    viewer.addModel(sdf, "sdf");
                    viewer.setStyle({}, { stick: { colorscheme: "Jmol" } });
                    viewer.zoomTo();
                    viewer.render();
                } else {
                    console.error("3Dmol.js not loaded or viewerRef missing.");
                }
            } catch (error) {
                console.error("Error initializing Viewer3D:", error);
            }
        };

        if (!window.$3Dmol) {
            const script = document.createElement("script");
            script.src = "https://3dmol.csb.pitt.edu/build/3Dmol-min.js";
            script.async = true;
            script.onload = initializeViewer;
            script.onerror = () => console.error("Failed to load 3Dmol.js.");
            document.body.appendChild(script);
        } else {
            initializeViewer();
        }
    }, [sdf, viewerRef]);

    return null; // We don't render anything here
}
