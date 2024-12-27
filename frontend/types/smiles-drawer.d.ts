declare module "smiles-drawer" {
  interface DrawOptions {
    width?: number;
    height?: number;
    // Add other options as needed
  }

  interface ParseOptions {
    // Define parse options if available
  }

  export class Drawer {
    constructor(options?: DrawOptions);
    draw(
      tree: any,
      target: string | HTMLCanvasElement,
      themeName?: string,
      infoOnly?: boolean
    ): void;
  }

  export function parse(
    smiles: string,
    successCallback: (tree: any) => void,
    errorCallback: (error: any) => void
  ): void;

  // If there are other exports, add them here
}
