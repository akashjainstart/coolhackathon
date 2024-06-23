import streamlit as st
from datetime import timedelta, datetime
from collections import deque
import time
import threading
import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate  # Ensure you have tabulate installed (pip install tabulate)

# Define job data (example data as provided)
job_data = [
    ["debug", 1, 30, 1, 4, 512, "72:00:00", "26:03:33", "ai_training_1", "ai_training_1.log", "ai_training_1.err", "ALL", "user@example.com"],
    ["compute", 8, 16, 1, 4, 512, "24:00:00", "38:41:20", "ai_training_2", "ai_training_2.log", "ai_training_2.err", "ALL", "user@example.com"],
    ["compute", 1, 1, 1, 32, 64, "24:00:00", "41:00:01", "ai_training_3", "ai_training_3.log", "ai_training_3.err", "ALL", "user@example.com"],
    ["gpu", 1, 2, 2, 8, 512, "24:00:00", "28:10:54", "ai_training_4", "ai_training_4.log", "ai_training_4.err", "ALL", "user@example.com"]
]


st.sidebar.title("SwitftCool AI: Control Tab")


st.title("SwiftCool AI")
st.write("Sample Tesla Data Center Confugration: 100 racks; 32 servers each rack.")

st.subheader("Upload AI workload data for model training.")
file = st.file_uploader("Upload AI job file for model training")
if file:
    st.write("The uploaded file isn't a valid workload file, please reload and try again.")
    time.sleep(1)
    st.write("Runninng results for sample dataset:")
    st.write("Jobs")
    st.write(job_data)

    # Define the job class
    class Job:
        def __init__(self, partition, gres_gpu, nodes, ntasks, cpus_per_task, memory_limit_G, time_limit, actual_run_time, job_name, output_log, error_log, mail_type, mail_user):
            self.partition = partition
            self.gres_gpu = gres_gpu
            self.nodes = nodes
            self.ntasks = ntasks
            self.cpus_per_task = cpus_per_task
            self.memory_limit_G = memory_limit_G
            self.time_limit = timedelta(hours=int(time_limit.split(':')[0]), minutes=int(time_limit.split(':')[1]), seconds=int(time_limit.split(':')[2]))
            self.actual_run_time = timedelta(hours=int(actual_run_time.split(':')[0]), minutes=int(actual_run_time.split(':')[1]), seconds=int(actual_run_time.split(':')[2]))
            self.job_name = job_name
            self.output_log = output_log
            self.error_log = error_log
            self.mail_type = mail_type
            self.mail_user = mail_user
            self.start_time = None
            self.end_time = None
            self.allocated_nodes = []
            self.free_time = None

    # Define the scheduler class
    class JobScheduler:
        def __init__(self):
            self.job_queue = deque()
            self.running_jobs = []
            self.completed_jobs = []
            self.racks = [(i, 32) for i in range(100)]  # Each rack has 32 nodes initially available
            self.lock = threading.Lock()
            self.df_rack_allocation = pd.DataFrame(columns=[f'Rack_{i}' for i in range(100)])

        def submit_job(self, job):
            self.job_queue.append(job)
            # st.write(f"Job {job.job_name} submitted")
            self.free_nodes()
            self.run_next_job()

        def run_next_job(self):
            if not self.job_queue:
                return

            job = self.job_queue.popleft()
            if self.allocate_resources(job):
                job.start_time = datetime.now()
                job.end_time = job.start_time + job.actual_run_time
                job.free_time = job.start_time + job.actual_run_time / 100
                self.running_jobs.append(job)
                st.write(f"Workload status at: " + str(datetime.now()))
                # st.write(f"Job {job.job_name} started at {job.start_time} on nodes {job.allocated_nodes}")
                self.update_rack_allocation(job)
                self.print_rack_allocation()
                threading.Timer(1, self.complete_job, args=[job]).start()

        def allocate_resources(self, job):
            required_nodes = job.nodes
            allocated_nodes = []

            for i, (rack, available_nodes) in enumerate(self.racks):
                if available_nodes > 0:
                    nodes_to_allocate = min(required_nodes, available_nodes)
                    self.racks[i] = (rack, available_nodes - nodes_to_allocate)
                    required_nodes -= nodes_to_allocate
                    allocated_nodes.extend([(rack, node) for node in range(32 - available_nodes, 32 - available_nodes + nodes_to_allocate)])

                    if required_nodes == 0:
                        job.allocated_nodes = allocated_nodes
                        return True

            for rack, node in allocated_nodes:
                for i, (r, available_nodes) in enumerate(self.racks):
                    if r == rack:
                        self.racks[i] = (r, available_nodes + 1)
                        break
            return False

        def free_nodes(self):
            time.sleep(0.2)
            with self.lock:
                current_time = datetime.now()
                for job in self.running_jobs[:]:
                    if job.free_time and job.free_time <= current_time:
                        self.running_jobs.remove(job)
                        for rack, node in job.allocated_nodes:
                            for i, (r, available_nodes) in enumerate(self.racks):
                                if r == rack:
                                    self.racks[i] = (r, available_nodes + 1)
                                    break
                        st.write(f"Freed nodes for job {job.job_name} at {current_time}")

        def complete_job(self, job):
            self.completed_jobs.append(job)
            st.write(f"Job {job.job_name} completed at {job.end_time}")
            self.send_notification(job)
            self.run_next_job()

        def update_rack_allocation(self, job):
            allocation_dict = {f'Rack_{rack}': 0 for rack in range(100)}
            for rack, node in job.allocated_nodes:
                allocation_dict[f'Rack_{rack}'] += 1
            new_row = pd.DataFrame(allocation_dict, index=[job.job_name])
            self.df_rack_allocation = pd.concat([self.df_rack_allocation, new_row])

        def print_rack_allocation(self):
            st.text(tabulate(self.df_rack_allocation, headers='keys', tablefmt='psql'))

        def send_notification(self, job):
            st.write(f"Sending notification to {job.mail_user} for job {job.job_name}")



    # Streamlit app
    st.title('Data Center Workload Simulation')

    # Instantiate the scheduler
    scheduler = JobScheduler()



    # Submit jobs to the scheduler
    for data in job_data:
        job = Job(*data)
        scheduler.submit_job(job)

    # Function to periodically free nodes
    def free_nodes_periodically():
        while scheduler.job_queue or scheduler.running_jobs:
            scheduler.free_nodes()
            time.sleep(100000)

    # Start a thread to free nodes periodically
    free_nodes_thread = threading.Thread(target=free_nodes_periodically)
    free_nodes_thread.daemon = True
    free_nodes_thread.start()

    # Display completed jobs
    st.subheader('Completed Jobs')
    for job in scheduler.completed_jobs:
        st.write(f"Job {job.job_name} completed at {job.end_time}")

    # Display rack allocation
    st.subheader('Rack Allocation')
    scheduler.print_rack_allocation()

    # Print message when all jobs are processed
    st.write("All jobs have been submitted and processed.")

    # Create a Plotly figure
    fig = go.Figure()

    # Add each column as a separate trace
    for col in scheduler.df_rack_allocation.columns:
        fig.add_trace(go.Scatter(
            y=scheduler.df_rack_allocation[col]*100/32,
            mode='lines',
            name=col
        ))

    # Update the layout
    fig.update_layout(
        title='Utilisation graph for Data center',
        xaxis_title='Time',
        yaxis_title='Utilisation percent',
        showlegend=True  # Hide legend if it becomes too cluttered
    )

    st.markdown('')


    # Display Plotly figure
    st.subheader('Utilisation Graph')
    st.plotly_chart(fig)