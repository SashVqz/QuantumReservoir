import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.linear_model import Ridge

class QuantumReservoirComputing:
    def __init__(self, nQubits=5, tau=0.4, vNodes=5, ridgeAlpha=1e-7):
        """
        Implementation of Quantum Reservoir Computing (QRC). references: 
        paper: Harnessing disordered ensemble quantum dynamics for machine learning by Fujii and Nakajima.
        """
        self.nQubits = nQubits
        self.tau = tau  # Time interval per input (Section II.C)
        self.vNodes = vNodes # Number of virtual nodes V (Section II.E)
        self.dt = tau / vNodes
        
        self.dev = qml.device("default.mixed", wires=nQubits)
        
        # Hamiltonian parameters (Section III)
        np.random.seed(42)
        # Jij distributed randomly from -J/2 to J/2. Using J=1.0.
        self.jCoeffs = np.random.uniform(-0.5, 0.5, (nQubits, nQubits))
        self.hCoeffs = np.random.uniform(-0.5, 0.5, nQubits)
        
        # Build Hamiltonian (Eq. 16)
        coeffs = []
        obs = []
        # Transverse field (Z field in paper's basis)
        for i in range(nQubits):
            coeffs.append(self.hCoeffs[i])
            obs.append(qml.PauliZ(i))
        # Interaction terms (XX interaction)
        for i in range(nQubits):
            for j in range(i + 1, nQubits):
                coeffs.append(self.jCoeffs[i, j])
                obs.append(qml.PauliX(i) @ qml.PauliX(j))
        
        self.hamiltonian = qml.Hamiltonian(coeffs, obs)
        self.readout = Ridge(alpha=ridgeAlpha, fit_intercept=True) # Intercept acts as bias (Eq. 12)

    def getInputState(self, uK):
        # Maps input signal to state vector (Eq. 6)
        theta = 2 * np.arcsin(np.sqrt(np.clip(uK, 0, 1)))
        state = np.array([np.cos(theta/2), np.sin(theta/2)])
        return np.outer(state, np.conj(state))

    def replaceFirstQubit(self, rhoFull, rhoIn):
        # Physical injection: rho_next = rho_in \otimes Tr_1(rho_old) (Eq. 7)
        dimRest = 2**(self.nQubits - 1)
        rhoReshaped = rhoFull.reshape((2, dimRest, 2, dimRest))
        rhoReduced = np.trace(rhoReshaped, axis1=0, axis2=2)
        return np.kron(rhoIn, rhoReduced)

    def evolveQNode(self, rhoInit, dt):
        # Time evolution (Eq. 1)
        @qml.qnode(self.dev)
        def circuit():
            qml.QubitDensityMatrix(rhoInit, wires=range(self.nQubits))
            qml.ApproxTimeEvolution(self.hamiltonian, dt, 1)
            return qml.state()
        return circuit()

    def measureObservable(self, rho, wire):
        # Expectation value <Zi> = Tr(rho * Zi)
        dimPre = 2**wire
        dimPost = 2**(self.nQubits - wire - 1)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        op = np.kron(np.kron(np.eye(dimPre), Z), np.eye(dimPost))
        return np.real(np.trace(rho @ op))

    def applyDynamics(self, rho, uK):
        # Temporal multiplexing via virtual nodes (Section II.E)
        features = []
        rhoIn = self.getInputState(uK)
        currentRho = rho
        
        for v in range(self.vNodes):
            if v == 0:
                # Injection at the start of interval tau
                stateToEvolve = self.replaceFirstQubit(currentRho, rhoIn)
            else:
                stateToEvolve = currentRho
            
            currentRho = self.evolveQNode(stateToEvolve, self.dt)
            
            # Extract rescaled signals (Eq. 9)
            for i in range(self.nQubits):
                xi = self.measureObservable(currentRho, i)
                features.append((xi + 1.0) / 2.0)
                
        return currentRho, np.array(features)

    def processInput(self, inputData):
        nSteps = len(inputData)
        totalNodes = self.nQubits * self.vNodes
        xMatrix = np.zeros((nSteps, totalNodes))
        
        # Initial state: vacuum
        rho = np.zeros((2**self.nQubits, 2**self.nQubits), dtype=complex)
        rho[0, 0] = 1.0
        
        for k in range(nSteps):
            rho, xK = self.applyDynamics(rho, inputData[k])
            xMatrix[k, :] = xK
            
        return xMatrix

    def train(self, trainInput, trainTarget, washout=100):
        # Linear readout training (Eq. 13)
        X = self.processInput(trainInput)
        self.readout.fit(X[washout:], trainTarget[washout:])
        return X

    def predict(self, testInput):
        # Reservoir output (Eq. 14)
        X = self.processInput(testInput)
        return self.readout.predict(X), X