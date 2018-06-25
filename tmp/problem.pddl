(define (problem blocks-1)
	(:domain blocksworld)
	(:objects b1 b2 b3 b4 b5 b6)
	(:init 
		(on-table b1)
		(on-table b2)
		(on-table b3)
		(on-table b4)
		(on-table b5)
		(clear b1)
		(clear b2)
		(clear b3)
		(clear b4)
		(clear b5)
		(arm-empty )
		(in-tower b4)
		(on-table b6)
		(clear b6)
	)
	(:goal (forall (?x) (in-tower ?x)))
)