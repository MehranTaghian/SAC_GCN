############## MEHRAN
# MELCO 1
rsync -a taghianj@melco.cs.ualberta.ca:/local/melco3/taghianj/SAC_GCN/Data /home/mehran/Documents/SAC_GCN

#MELCO 2
rsync -a taghianj@melco2.cs.ualberta.ca:/home/taghianj/Documents/SAC_GCN/Data /home/mehran/Documents/SAC_GCN

# MEHRAN
rsync -a taghianj@mehran.cs.ualberta.ca:/home/taghianj/Documents/SAC_GCN/Data /home/mehran/Documents/SAC_GCN

# cedar
rsync -a taghianj@cedar.computecanada.ca:/home/taghianj/scratch/SAC_GCN/Data /home/mehran/Documents/SAC_GCN/Data/cedar

##################################### TAGHIANJ
# MELCO 1
rsync -a taghianj@melco.cs.ualberta.ca:/local/melco3/taghianj/SAC_GCN/Data /home/taghianj/Documents/SAC_GCN
rsync -a taghianj@melco.cs.ualberta.ca:/local/melco3/taghianj/SAC_GCN/Data/Walker2dBroken-v2/ /home/taghianj/Documents/SAC_GCN/Data/Walker2dBroken-v2/


#MELCO 2
rsync -a taghianj@melco2.cs.ualberta.ca:/home/taghianj/Documents/SAC_GCN/Data /home/taghianj/Documents/SAC_GCN
rsync -a taghianj@melco2.cs.ualberta.ca:/home/taghianj/Documents/SAC_GCN/Data/Walker2dBroken-v2/ /home/taghianj/Documents/SAC_GCN/Data/Walker2dBroken-v2/

# GOOGLE DRIVE
rsync -a /home/taghianj/Documents/SAC_GCN/Data/ /home/taghianj/google-drive/SAC-GCN/


# cedar
rsync -a taghianj@cedar.computecanada.ca:/home/taghianj/scratch/SAC_GCN/Data/* /home/taghianj/Documents/SAC_GCN/Data/cedar

# beluga
rsync -a taghianj@beluga.computecanada.ca:/home/taghianj/scratch/SAC_GCN/Data/* /home/taghianj/Documents/SAC_GCN/Data/beluga

##################################### MELCO
# cedar
rsync -a taghianj@cedar.computecanada.ca:/home/taghianj/scratch/SAC_GCN/Data/* /local/melco3/taghianj/SAC_GCN/Data/cedar

# beluga
rsync -a taghianj@beluga.computecanada.ca:/home/taghianj/scratch/SAC_GCN/Data/* /local/melco3/taghianj/SAC_GCN/Data/beluga
