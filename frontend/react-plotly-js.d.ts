// src/types/react-plotly-js.d.ts
declare module "react-plotly.js" {
  import { Component } from "react";
  import Plotly from "plotly.js";

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    onInitialized?: (
      figure: Readonly<PlotParams>,
      graphDiv: Readonly<HTMLDivElement>
    ) => void;
    onUpdate?: (
      figure: Readonly<PlotParams>,
      graphDiv: Readonly<HTMLDivElement>
    ) => void;
    onPurge?: (
      figure: Readonly<PlotParams>,
      graphDiv: Readonly<HTMLDivElement>
    ) => void;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    debug?: boolean;
    onClick?: () => void;
    onHover?: () => void;
    onUnhover?: () => void;
    onSelected?: () => void;
    divId?: string;
  }

  export default class Plot extends Component<PlotParams> {}
}
