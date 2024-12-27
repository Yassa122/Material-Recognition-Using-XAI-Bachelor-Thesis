// types/molecule.ts
export interface Atom {
  id: string;
  element: string;
  x: number;
  y: number;
}

export interface Bond {
  id: string;
  from: string; // Atom ID
  to: string; // Atom ID
  type: number; // 1 for single, 2 for double, 3 for triple
}

export interface Molecule {
  atoms: Atom[];
  bonds: Bond[];
}
