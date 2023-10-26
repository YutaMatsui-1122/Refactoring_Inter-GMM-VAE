from parameter_setting import *
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import cohen_kappa_score as Kappa


class CommunicationField():
    def __init__(self, agentA, agentB):
        self.agentA = agentA
        self.agentB = agentB
        self.truth_w = self.set_truth_w()

    def set_truth_w(self):
        truth_w = []
        for (data, w, index) in self.agentA.all_loader:
            truth_w.append(w)
        truth_w = torch.cat(truth_w, dim=0).numpy()
        return truth_w

    def symbol_emergence(self):
        for m in range(mi_iter):
            self.agentA.vae(m)
            self.agentB.vae(m)
            self.MH_naming_game()
            self.agentA.set_vae_prior()
            self.agentB.set_vae_prior()

    def MH_naming_game(self):
        self.agentA.initialize_parameters()
        self.agentB.initialize_parameters()
        for i in range(mh_iter):
            proposed_w = self.agentA.propose()
            self.agentB.accept_or_reject(proposed_w)
            self.agentB.update()
            proposed_w = self.agentB.propose()
            self.agentA.accept_or_reject(proposed_w)
            self.agentA.update()
            if i % 10 == 0 or i == mh_iter - 1:
                print("MH iter:", i,"ARI_A:", ARI(self.agentA.w, self.truth_w), "ARI_B:", ARI(self.agentB.w, self.truth_w), "Kappa:", Kappa(self.agentA.w, self.agentB.w))