package repositories

import models.Task

import scala.collection.mutable
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

class TaskRepository {
  private val tasks = mutable.Map[Long, Task]()
  private var currentId: Long = 0

  def create(task: Task): Future[Task] = Future {
    currentId += 1
    val newTask = task.copy(id = currentId)
    tasks(currentId) = newTask
    newTask
  }

  def findAll(): Future[Seq[Task]] = Future {
    tasks.values.toSeq
  }

  def findById(id: Long): Future[Option[Task]] = Future {
    tasks.get(id)
  }

  def update(task: Task): Future[Option[Task]] = Future {
    tasks.get(task.id).map { _ =>
      tasks(task.id) = task
      task
    }
  }

  def delete(id: Long): Future[Option[Task]] = Future {
    tasks.remove(id)
  }
}